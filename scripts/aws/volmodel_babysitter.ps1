# Autonomous Gen-2 babysitter for the MULTI-HORIZON volatility-forecast cloud run.
#
# Ports scripts/lbr6_search_babysitter.ps1 and the AWS-spot operating model
# from the earlier Pokerbot implementation.
# Generalized from "one box per run" to "one box per HORIZON": each horizon H trains the
# walk-forward vol forecast on its own spot box and uploads a result shard to
#   s3://BUCKET/PREFIX/results/h{H}/shard.json
# The babysitter polls S3 for each horizon's shard + EC2 for that horizon's live box (tag:Run),
# and relaunches reclaimed/missing horizons. Done horizons are skipped (idempotent: the launcher's
# on-box driver self-skips if the shard already exists, and the babysitter never relaunches a horizon
# whose shard is present). It enforces a wall-clock + launch-count budget and tears every box down on
# any exit path.
#
# Run from YOUR terminal (your AWS creds):
#   powershell -ExecutionPolicy Bypass -File scripts\aws\volmodel_babysitter.ps1 -RunId volrun1 -Horizons 5,15,30
#
# The 3 Gen-2 fixes are preserved (see inline comments tagged FIX-A/B/C):
#   FIX-A relaunch-storm guard: parse 'LAUNCHED i-<id>' from the launcher, 300s boot grace, verify the
#         box by instance-id (strongly consistent) not the laggy tag filter, and do NOT count a failed
#         launch (MaxSpotInstanceCountExceeded / capacity) against the launch cap.
#   FIX-B timezone fix: parse EC2 LaunchTime WITH its offset via [datetimeoffset]::Parse(...).UtcDateTime
#         so uptime is not inflated by the local UTC offset (the phantom +120m bug).
#   FIX-C stale-marker cleanup at startup: remove this run's prior FAIL_* / shard.json so the dashboard
#         reflects only this run -- but only for horizons NOT yet done (a present shard means done, keep it).
#
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).

param(
  [string]$RunId      = "volrun1",
  # Six-horizon fleet. h=1440 is excluded because its available training
  # history does not meet the minimum-row gate. One box per horizon; tf=H*60.
  [int[]]$Horizons    = @(5, 15, 30, 60, 240, 480),
  [string]$Arch       = "arm64",        # Graviton (c7g) is cheapest compute-optimized per vCPU.
  [int]$MaxSec        = 16 * 3600,      # wall-clock budget across ALL horizons (worst-case h=5 + margin)
  [int]$MaxLaunch     = 18,             # total spot launches across 6 horizons (storm cap; ~3/horizon)
  [int]$PollSec       = 180,
  [switch]$OnDemand                     # use ON-DEMAND instead of spot (guaranteed capacity; ~EUR3 more/run, no reclaim churn)
)

# BOX LAYOUT -- right-sized small boxes for the thread-flat LightGBM fit.
# Larger boxes add little fit throughput and waste cores.
# One horizon per box, the box sized to the few threads its fit uses efficiently, fit pinned via
# OPT_NUM_THREADS (= --opt-num-threads, passed below). Wall-clock floor = the SLOWEST single horizon
# (h=5, most rows; months x trials are SEQUENTIAL) -- shrinking boxes cuts COST, not that floor.
#   * h=5  : c7g.xlarge (4 vCPU)  -- most rows; keep 3 fit-threads + IO/BLAS headroom. The long pole.
#   * h=15 : c7g.xlarge (4 vCPU)  -- BYTE-IDENTICAL: launched WITHOUT --opt-num-threads => native -1
#                                     => the fit uses all 4 box cores (sacred model preserved).
#   * h=30,60,240,480 : c7g.large (2 vCPU) -- fit pinned to 2 threads; datasets ~1/H of h=5.
# arm64 c7g (Graviton3) replaces c6g: ~same $/vCPU, faster, and the smallest size (.large=2 vCPU)
# is the right granularity for a 2-thread-pinned coarse horizon. Concurrent total = 4+4+2+2+2+2 = 16
# vCPU across the 6 boxes (all in ONE wave; trivially under any spot vCPU budget).
# Spot capacity for one (type,size,AZ) pool can run dry (InsufficientInstanceCapacity). Each horizon
# falls back across EQUIVALENT non-burstable arm64 types (same size) so it has SEVERAL spot pools, not
# one; the per-horizon index advances on every capacity failure -> the next attempt hits a different pool.
# The chains are DATA-DRIVEN from spotpick (cheapest-per-family from the EC2 cost catalog, t-series
# excluded; path supplied through $env:SPOTPICK). If spotpick is absent the built-in fallback
# (c7g->c6g->m6g) is used. NOTE: spotpick ranks by $/vCPU, so c6g (Graviton2) may lead c7g (Graviton3) --
# ~8% cheaper but ~15% slower/core; fine here (cost trivial, capacity is the constraint, and older-gen
# usually has better spot availability anyway).
if ([string]::IsNullOrWhiteSpace($env:SPOTPICK)) { throw "SPOTPICK must point to spotpick.py" }
$SpotpickPy = $env:SPOTPICK.Trim()
if (-not (Test-Path -LiteralPath $SpotpickPy -PathType Leaf)) {
  throw "SPOTPICK must point to an existing spotpick.py"
}
$PyForPick  = if ($env:QUANT_PYTHON) { $env:QUANT_PYTHON } else { Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe" }
function Get-SpotChain([int]$vcpus, [int]$n, [string[]]$fallback) {
  if ((Test-Path $SpotpickPy) -and (Test-Path $PyForPick)) {
    try {
      $od = if ($OnDemand) { @("--on-demand") } else { @() }
      $raw = (& $PyForPick $SpotpickPy chain --vcpus $vcpus --arch $Arch --n $n @od 2>$null)
      if ($LASTEXITCODE -eq 0 -and $raw) {
        $arr = @(($raw | Out-String).Trim() -split "\s+" | Where-Object { $_ })
        if ($arr.Count -gt 0) { Write-Host "  [spotpick ${vcpus}vCPU/$Arch chain: $($arr -join ', ')]"; return $arr }
      }
    } catch {}
  }
  Write-Host "  [spotpick unavailable -> built-in fallback for ${vcpus}vCPU: $($fallback -join ', ')]" -ForegroundColor DarkYellow
  return $fallback
}
# --n 99 = use the FULL breadth of equivalent families spotpick has (~22 at .large / ~21 at .xlarge),
# cheapest-first, so the failover spans ~20+ spot pools instead of 4. Cheap families (c6g/c7g/m6g...)
# are tried first; the pricier memory/storage tail (r/x/i Graviton) is only reached if everything
# cheaper is dry -- a true last resort, still pennies at .large/.xlarge.
$ChainXL  = Get-SpotChain 4 99 @("c7g.xlarge", "c6g.xlarge", "m6g.xlarge", "c8g.xlarge", "m7g.xlarge")   # 4 vCPU (h5, h15)
$ChainL   = Get-SpotChain 2 99 @("c7g.large", "c6g.large", "m6g.large", "c8g.large", "m7g.large")        # 2 vCPU (h30/60/240/480)
$ChainFor = @{ 5 = $ChainXL; 15 = $ChainXL; 30 = $ChainL; 60 = $ChainL; 240 = $ChainL; 480 = $ChainL }
# Per-horizon LightGBM fit thread pin (OPT_NUM_THREADS). h=15 is ABSENT on purpose -> the launch line
# omits --opt-num-threads for it -> native -1 -> byte-identical. h=5 gets 3 (leaves 1 core for BLAS/IO
# on its 4-vCPU box); the coarse horizons get 2 (their 2-vCPU box). Gate: threads <= box vCPUs, always.
$OptThreadsFor = @{ 5 = 3; 30 = 2; 60 = 2; 240 = 2; 480 = 2 }
$DefaultInstance = "c7g.large"
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")   # repo root (scripts/aws -> repo)

# --- constants from env (keep out of source; defaults match the porting findings) ----------------
$REGION   = if ($env:QUANT_REGION) { $env:QUANT_REGION } else { "eu-central-1" }
if ([string]::IsNullOrWhiteSpace($env:QUANT_BUCKET)) { throw "QUANT_BUCKET must be set" }
$BUCKET   = $env:QUANT_BUCKET.Trim()
$PREFIX   = if ($env:QUANT_PREFIX) { $env:QUANT_PREFIX } else { "volmodel/v1" }
# Launcher is configurable so this is robust to the sibling agent's final filename.
$LAUNCHER = if ($env:QUANT_LAUNCHER) { $env:QUANT_LAUNCHER } else { "scripts\aws\launch_volmodel_spot.py" }
$PYTHON   = if ($env:QUANT_PYTHON) { $env:QUANT_PYTHON } else { ".venv\Scripts\python.exe" }

# tag:Run is unique per (run, horizon) so each horizon's box is found/terminated independently.
function RunTag([int]$h) { return "$RunId-h$h" }
function ShardKey([int]$h) { return "$PREFIX/results/h$h/shard.json" }

$start    = Get-Date
$launches = 0
$CodeUploaded = $false   # upload the code tarball ONCE per run; reuse (--skip-code-upload) thereafter
# per-horizon tracked instance id (verified by id, not the laggy tag filter) + last-seen box id.
$tracked  = @{}
$prevBox  = @{}
foreach ($h in $Horizons) { $tracked[$h] = ""; $prevBox[$h] = "" }

function Test-ShardDone([int]$h) {
  aws s3 ls "s3://$BUCKET/$(ShardKey $h)" --region $REGION 2>$null | Out-Null
  return $?
}
function Get-Boxes([int]$h, [string]$states) {
  $tag = RunTag $h
  $r = (aws ec2 describe-instances --region $REGION `
        --filters "Name=tag:Run,Values=$tag" "Name=instance-state-name,Values=$states" `
        --query "Reservations[].Instances[].[InstanceId,LaunchTime]" --output text 2>$null)
  if ($r) { return (($r -join " ") -replace "\s+", " ").Trim() } else { return "" }
}
function Get-StateById([string]$id) {
  # FIX-A: strongly-consistent check by instance id (the tag:Run filter lags ~minutes for a fresh
  # spot box, which caused a relaunch storm -- track the launched id and verify it directly).
  if (-not $id) { return "" }
  $s = (aws ec2 describe-instances --region $REGION --instance-ids $id `
        --query "Reservations[].Instances[].State.Name" --output text 2>$null)
  if ($s) { return $s.Trim() } else { return "" }
}
function Stop-AllRuns {
  # Terminate every running/pending/stopping box across ALL horizons of this run. Called on every
  # exit path (done / time cap / launch cap) so a dead babysitter never leaves billing boxes.
  foreach ($h in $Horizons) {
    $ids = (Get-Boxes $h "running,pending,stopping")
    if ($ids) {
      $idList = @($ids.Split(" ") | Where-Object { $_ -like "i-*" })
      if ($idList.Count) {
        aws ec2 terminate-instances --instance-ids $idList --region $REGION | Out-Null
        Write-Host "terminated (h$h): $idList"
      }
    }
  }
}

$capH = [int]($MaxSec / 3600)
$hlist = ($Horizons -join ",")
Write-Host "=== volmodel babysitter: run=$RunId horizons=[$hlist]m  caps: ${capH}h / $MaxLaunch launches ==="
Write-Host "    bucket=s3://$BUCKET/$PREFIX  region=$REGION  launcher=$LAUNCHER"

# FIX-C: one-time cleanup of this run's stale FAIL_* + shard.json from a PRIOR aborted attempt, so the
# dashboard reflects only this run. A present shard means the horizon is DONE -> keep it (resume/skip).
foreach ($h in $Horizons) {
  if (-not (Test-ShardDone $h)) {
    aws s3 rm "s3://$BUCKET/$PREFIX/results/h$h/" --region $REGION --recursive `
      --exclude "*" --include "FAIL_*" --include "shard.json" 2>$null | Out-Null
  }
}

while ($true) {
  $el = [int]((Get-Date) - $start).TotalSeconds

  # ---- assess every horizon: done? live box? -----------------------------------------------------
  $allDone   = $true
  $needsWork = @()   # horizons not done and with no live box -> candidates for (re)launch
  foreach ($h in $Horizons) {
    $done = Test-ShardDone $h
    if (-not $done) { $allDone = $false }

    $live = Get-Boxes $h "running,pending"
    if (-not $live -and $tracked[$h]) {
      # FIX-A: tag filter not caught up yet -> verify the box we launched is still alive by id.
      $st = Get-StateById $tracked[$h]
      if ($st -eq "running" -or $st -eq "pending") { $live = $tracked[$h] }
      else { $tracked[$h] = "" }   # reclaimed/terminated -> allow a relaunch
    }

    $boxId = ""; $up = ""
    if ($live) {
      $boxId = (@($live.Split(" ") | Where-Object { $_ -like "i-*" }) -join ",")
      if ($live -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
        # FIX-B: parse LaunchTime WITH offset via DateTimeOffset, else the UTC stamp is read as local
        # and uptime is inflated by the local offset (the phantom +120m bug).
        try {
          $mins = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes)
          $up = " (up ${mins}m)"
        } catch {}
      }
    }

    $state = if ($done) { "done" } elseif ($boxId) { "live $boxId$up" } else { "idle" }
    Write-Host ("[t+${el}s] h$h : $state")

    if ($prevBox[$h] -and $boxId -and $boxId -ne $prevBox[$h]) {
      Write-Host ("          h$h BOX CHANGED: {0} -> {1} (spot reclaim + relaunch)" -f $prevBox[$h], $boxId) -ForegroundColor Yellow
    }
    if ($boxId) { $prevBox[$h] = $boxId }

    if (-not $done -and -not $boxId) { $needsWork += $h }
  }
  $nDone = (@($Horizons | Where-Object { Test-ShardDone $_ })).Count
  Write-Host ("[t+${el}s] summary: $nDone/$($Horizons.Count) horizons done, launches=$launches/$MaxLaunch")

  # ---- exit conditions ---------------------------------------------------------------------------
  if ($allDone)        { Write-Host "DONE: all horizons have a result shard"; Stop-AllRuns; break }
  if ($el -ge $MaxSec) { Write-Host "STOP: wall-clock cap reached"; Stop-AllRuns; break }

  # ---- (re)launch ALL needy horizons THIS cycle (each tries its whole type chain), then ONE grace --
  if ($needsWork.Count -eq 0) {
    # All not-done horizons currently have a live box -> just wait.
    Start-Sleep -Seconds $PollSec
    continue
  }
  if ($launches -ge $MaxLaunch) {
    Write-Host "STOP: max launches ($MaxLaunch) hit -- giving up on remaining horizons"
    Stop-AllRuns; break
  }
  Write-Host ("[t+${el}s] launching ALL idle horizons this wave: $($needsWork -join ', ')")
  $anyLaunched = $false
  foreach ($h in $needsWork) {
    if ($launches -ge $MaxLaunch) { Write-Host "  max launches ($MaxLaunch) hit -- remaining horizons next round"; break }
    $tf = $h * 60
    $chain  = if ($ChainFor.ContainsKey($h)) { $ChainFor[$h] } else { @($DefaultInstance) }
    $itypes = ($chain -join ",")   # the launcher fails over these INSTANTLY on InsufficientInstanceCapacity
    # [string[]] casts are REQUIRED: a single-element array from `if` unwraps to a scalar string, and
    # splatting a string char-splits it (--skip-code-upload -> '-','-','s','k',...). The cast keeps it
    # an array so @mktArgs / @skipArgs splat as one argument.
    [string[]]$mktArgs = if ($OnDemand) { @("--on-demand") } else { @() }
    $mktDisp = if ($OnDemand) { "on-demand" } else { "spot" }
    # Upload the LIVE code tarball ONCE per babysitter run; the rest of the wave reuses it
    # (--skip-code-upload) -> no 114MB re-upload per horizon AND every box runs the SAME code.
    [string[]]$skipArgs = if ($CodeUploaded) { @("--skip-code-upload") } else { @() }
    # Per-horizon LightGBM fit thread pin. h=15 NOT in $OptThreadsFor -> omit --opt-num-threads -> native -1.
    $optArgs = @()
    $optDisp = "native -1 (h15 byte-identical)"
    if ($OptThreadsFor.ContainsKey($h)) {
      $optArgs = @("--opt-num-threads", $OptThreadsFor[$h])
      $optDisp = "OPT_NUM_THREADS=$($OptThreadsFor[$h])"
    }
    Write-Host "  h$h launching ($itypes / $Arch $mktDisp, tf=${tf}s, $optDisp; launch $($launches + 1)/$MaxLaunch)"
    $out = (& $PYTHON $LAUNCHER `
              --tf $tf --no-transvol `
              --arch $Arch --instance-types $itypes `
              @optArgs @mktArgs @skipArgs `
              --run-id (RunTag $h) `
              --bucket $BUCKET --prefix $PREFIX --region $REGION 2>&1)
    $out | ForEach-Object { Write-Host $_ }
    $txt = ($out | Out-String)
    $CodeUploaded = $true   # the launcher (re)built+uploaded the tarball on this first call of the run

    if ($txt -match "LAUNCHED (i-[0-9a-f]+)") {
      # real box started (the launcher already failed over types on capacity). Track + count it.
      $tracked[$h] = $Matches[1]; $launches++; $anyLaunched = $true
      Write-Host "  h$h tracking $($tracked[$h])"
    } elseif ($txt -match "LAUNCH_QUOTA") {
      # account vCPU/spot LIMIT (NOT a per-type capacity miss) -> other types won't help; just wait.
      Write-Host "  h$h QUOTA-limited -> retry after running boxes free quota (launch NOT counted)" -ForegroundColor Yellow
    } else {
      # CAPACITY_EXHAUSTED (all chain types dry) or a launcher error -> retry next cycle.
      Write-Host "  h$h no capacity across [$itypes] (launch NOT counted) -> retry next cycle" -ForegroundColor DarkYellow
    }
  }
  # ONE grace for the whole wave: 300s boot+tag-propagation if anything launched, else a short poll.
  if ($anyLaunched) {
    Write-Host "  (wave launched; 300s boot grace before re-evaluating)"
    Start-Sleep -Seconds 300
  } else {
    Start-Sleep -Seconds $PollSec
  }
}

# ---- final report ---------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== volmodel babysitter results ($RunId) ==="
foreach ($h in $Horizons) {
  if (Test-ShardDone $h) {
    Write-Host ("  h$h DONE  -> aws s3 cp s3://$BUCKET/$(ShardKey $h) -") -ForegroundColor Green
  } else {
    Write-Host ("  h$h INCOMPLETE (no shard; check FAIL_* / get-console-output)") -ForegroundColor Yellow
  }
}
Write-Host "BABYSITTER EXIT after ${el}s, $launches launches"
