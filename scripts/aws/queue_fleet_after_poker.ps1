# Queue the vol-model fleet to launch AUTOMATICALLY after the poker run_rnr1 eval finishes.
#
# WHY a waiter (not just "run both now"): the poker box is an m7g.16xlarge (64 vCPU) and the vol
# fleet is 6x c7g (16 vCPU), BOTH arm64 spot in eu-central-1 -> they share one Standard-family spot
# vCPU quota (64+16=80), so running them back-to-back avoids a quota collision.
#
# WHAT this does, in order:
#   1. Polls until the poker eval finishes: the DONE marker appears
#      (s3://POKER_BUCKET/pokerbot/prod6000/eval_suite/run_rnr1/eval_suite_DONE.marker),
#      OR the poker box has been CONTINUOUSLY absent for -BoxGoneGraceMin minutes of SUCCESSFUL polls
#      (cap/give-up fallback). An AWS call that ERRORS is never read as "finished".
#   2. Waits for the poker spot box to fully terminate (frees the shared spot vCPU quota).
#   3. (unless -SkipPreflight) Runs ONE on-demand arm64 h=480 pre-flight (cheap, ~minutes). This is
#      the FIRST-EVER arm64 run, so it validates the whole path end-to-end (IAM, data staging in S3,
#      Graviton build, train->shard) on 1 box. If it does NOT produce a shard, the chain STOPS and
#      does NOT launch the fleet. On success it purges the pre-flight's result shard AND its
#      checkpoint dir so the real fleet re-runs h=480 at FULL budget (the driver restores+skips
#      months from checkpoints/h{H}/ -> leaving the 10-trial checkpoints would contaminate h480).
#   4. Launches the fleet babysitter (volmodel_babysitter.ps1 -RunId <VolRunId>).
#
# RUN THIS IN A NEW PowerShell window (your AWS creds), leave it; the poker babysitter keeps running
# in its own window untouched:
#   powershell -ExecutionPolicy Bypass -File scripts\aws\queue_fleet_after_poker.ps1
# Bare babysitter only (skip the arm64 validation): add  -SkipPreflight
#
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII into string-breaking bytes).

param(
  [string]$PokerRun         = "run_rnr1",
  [string]$PokerBucket      = $env:POKER_BUCKET,
  [string]$PokerPrefix      = "pokerbot/prod6000/eval_suite",
  [string]$VolRunId         = "volrun1",
  [switch]$SkipPreflight,
  [int]$PollSec             = 60,
  [int]$BoxGoneGraceMin     = 30,   # no marker + poker box absent this long (successful polls) -> finished
  [int]$PreflightTimeoutMin = 75,   # max wait for the pre-flight shard before declaring failure
  [int]$PokerDrainMin       = 12    # max wait for the poker box to terminate before launching fleet
)

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")   # repo root (scripts/aws -> repo)

# Vol-fleet env (defaults match volmodel_babysitter.ps1 / volmodel_env.py).
if (-not $env:QUANT_REGION) { $env:QUANT_REGION = "eu-central-1" }
if ([string]::IsNullOrWhiteSpace($env:QUANT_BUCKET)) { throw "QUANT_BUCKET must be set" }
if ([string]::IsNullOrWhiteSpace($PokerBucket)) { throw "PokerBucket or POKER_BUCKET must be set" }
if (-not $env:QUANT_PREFIX) { $env:QUANT_PREFIX = "volmodel/v1" }
$REGION = $env:QUANT_REGION
$QBUCKET = $env:QUANT_BUCKET.Trim()
$PokerBucket = $PokerBucket.Trim()
$QPREFIX = $env:QUANT_PREFIX
$PY = ".venv\Scripts\python.exe"
$LAUNCHER = "scripts\aws\launch_volmodel_spot.py"
$PokerMarker = "$PokerPrefix/$PokerRun/eval_suite_DONE.marker"
$PreflightShardLs = "s3://$QBUCKET/$QPREFIX/results/h480/shard.json"

if (-not (Test-Path $PY)) {
  Write-Host "STOP: venv python not found at $PY -- run from the repo with the .venv present." -ForegroundColor Red
  exit 3
}

$logDir = "vol_arb\output"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
try { Start-Transcript -Path "$logDir\queue_fleet_after_poker.log" -Append | Out-Null } catch {}

function Now { return (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }

# Test-* return $true ONLY on a successful (exit 0) 'object found'. An errored AWS call returns $false
# = "not found yet", which is the SAFE direction (keep waiting), so these need no ERR sentinel.
function Test-PokerDone {
  aws s3 ls "s3://$PokerBucket/$PokerMarker" --region $REGION 2>$null | Out-Null
  return ($LASTEXITCODE -eq 0)
}
function Test-PreflightShard {
  aws s3 ls $PreflightShardLs --region $REGION 2>$null | Out-Null
  return ($LASTEXITCODE -eq 0)
}
# Get-* distinguish "confirmed empty" (exit 0, no rows) from "AWS call FAILED" (exit != 0 -> "ERR").
# describe-instances returns exit 0 even with zero matches, so a non-zero exit is a genuine call error.
function Get-PokerBox {
  $r = (aws ec2 describe-instances --region $REGION `
        --filters "Name=tag:Run,Values=$PokerRun" "Name=instance-state-name,Values=running,pending" `
        --query "Reservations[].Instances[].InstanceId" --output text 2>$null)
  if ($LASTEXITCODE -ne 0) { return "ERR" }
  if ($r) { return ($r.Trim() -replace "\s+", " ") } else { return "" }
}
function Get-StateById([string]$id) {
  if (-not $id) { return "" }
  $s = (aws ec2 describe-instances --region $REGION --instance-ids $id `
        --query "Reservations[].Instances[].State.Name" --output text 2>$null)
  if ($LASTEXITCODE -ne 0) { return "ERR" }
  if ($s) { return $s.Trim() } else { return "" }
}

Write-Host "=== queue_fleet_after_poker  ($(Now)) ==="
Write-Host "  WAIT FOR : poker '$PokerRun'  (marker s3://$PokerBucket/$PokerMarker)"
Write-Host "  THEN     : $(if ($SkipPreflight) {'launch fleet babysitter (NO pre-flight)'} else {'arm64 pre-flight (gated) -> fleet babysitter'})"
Write-Host "  FLEET    : volmodel_babysitter.ps1 -RunId $VolRunId   (region $REGION, bucket $QBUCKET)"
Write-Host ""

# ---- 1. wait for poker to finish ------------------------------------------------------------------
$goneSince = $null
while ($true) {
  if (Test-PokerDone) { Write-Host "[$(Now)] poker DONE marker present -> proceeding."; break }
  $box = Get-PokerBox
  if ($box -eq "ERR") {
    Write-Host "[$(Now)] AWS poll FAILED (unreachable/creds?) -- NOT counting toward give-up; retrying."
    # Deliberately do NOT touch $goneSince: an errored call must never look like 'poker finished'.
  } elseif ($box) {
    $goneSince = $null
    Write-Host "[$(Now)] poker still running (box $box); no marker yet."
  } else {
    if (-not $goneSince) { $goneSince = Get-Date }
    $goneMin = [int]((Get-Date) - $goneSince).TotalMinutes
    Write-Host "[$(Now)] poker: no marker AND no live box for ${goneMin}m (give-up fallback at ${BoxGoneGraceMin}m)."
    if ($goneMin -ge $BoxGoneGraceMin) {
      Write-Host "[$(Now)] poker box continuously absent ${goneMin}m with no marker -> treating poker as finished."
      break
    }
  }
  Start-Sleep -Seconds $PollSec
}

# ---- 2. wait for the poker box to fully terminate (free the shared spot vCPU quota) ----------------
$drainStart = Get-Date
while ($true) {
  $box = Get-PokerBox
  if ($box -eq "ERR") {
    Write-Host "[$(Now)] AWS poll FAILED during drain -- retrying; will NOT launch while AWS is unreachable."
    Start-Sleep -Seconds 30
    continue
  }
  if (-not $box) { Write-Host "[$(Now)] poker box terminated (spot quota freed)."; break }
  $waited = [int]((Get-Date) - $drainStart).TotalMinutes
  if ($waited -ge $PokerDrainMin) {
    Write-Host "[$(Now)] WARNING: poker box ($box) still present after ${waited}m. Proceeding anyway (on-demand pre-flight is quota-independent; if the SPOT fleet hits MaxSpotInstanceCountExceeded the babysitter backs off + retries without counting it)." -ForegroundColor Yellow
    break
  }
  Write-Host "[$(Now)] waiting for poker box ($box) to terminate... (${waited}m)"
  Start-Sleep -Seconds 30
}

# ---- 3. arm64 pre-flight gate (skippable) ---------------------------------------------------------
# Purge BOTH the result shard AND the per-horizon checkpoint dir for h480: the driver restores+skips
# completed months from checkpoints/h{H}/, so a leftover 10-trial pre-flight checkpoint would make the
# real fleet REUSE the smoke months instead of training h480 at full 200-trial budget.
function Clear-H480Artifacts {
  aws s3 rm "s3://$QBUCKET/$QPREFIX/results/h480/" --region $REGION --recursive 2>$null | Out-Null
  aws s3 rm "s3://$QBUCKET/$QPREFIX/checkpoints/h480/" --region $REGION --recursive 2>$null | Out-Null
}

if (-not $SkipPreflight) {
  if (Test-PreflightShard) {
    Write-Host "[$(Now)] a stale h480 shard exists -> clearing results/ + checkpoints/ for a clean pre-flight."
  }
  Clear-H480Artifacts
  if (Test-PreflightShard) {
    Write-Host "[$(Now)] PRE-FLIGHT ABORT: could not delete the stale h480 shard (perms?). NOT launching the fleet." -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch {}
    exit 4
  }
  Write-Host "[$(Now)] launching arm64 pre-flight: on-demand c7g.large h=480, --max-trials 10 --patience 5 (run-id preflight-arm64)."
  $out = (& $PY $LAUNCHER `
            --horizon 480 --arch arm64 --instance-type c7g.large `
            --opt-num-threads 2 --on-demand --max-trials 10 --patience 5 `
            --run-id preflight-arm64 `
            --bucket $QBUCKET --prefix $QPREFIX --region $REGION 2>&1)
  $out | ForEach-Object { Write-Host $_ }
  $txt = ($out | Out-String)
  $pfId = ""
  if ($txt -match "(?m)^LAUNCHED (i-[0-9a-f]+)") { $pfId = $Matches[1] }
  if (-not $pfId) {
    Write-Host "[$(Now)] PRE-FLIGHT FAILED TO LAUNCH (no 'LAUNCHED i-...' from the launcher; likely IAM/quota/permissions). NOT launching the fleet." -ForegroundColor Red
    Write-Host "  Fix the cause, then re-run this script (or run with -SkipPreflight only if you accept the risk)."
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
  }
  Write-Host "[$(Now)] pre-flight box $pfId launched; polling for shard (timeout ${PreflightTimeoutMin}m). It self-terminates on completion."
  $pfStart = Get-Date
  $pfOk = $false
  while ($true) {
    if (Test-PreflightShard) { $pfOk = $true; break }
    $st = Get-StateById $pfId
    $mins = [int]((Get-Date) - $pfStart).TotalMinutes
    if ($st -eq "ERR") {
      Write-Host "[$(Now)] AWS poll FAILED while watching the pre-flight box -- retrying."
    } elseif ($st -ne "running" -and $st -ne "pending") {
      # Box gone: give S3 a moment, then check the shard once more (terminate races the final upload).
      Start-Sleep -Seconds 20
      if (Test-PreflightShard) { $pfOk = $true }
      Write-Host "[$(Now)] pre-flight box state='$st' (terminated) after ${mins}m; shard present=$pfOk."
      break
    } elseif ($mins -ge $PreflightTimeoutMin) {
      Write-Host "[$(Now)] pre-flight TIMEOUT after ${mins}m (box still '$st', no shard)." -ForegroundColor Yellow
      break
    } else {
      Write-Host "[$(Now)] pre-flight running ($pfId, ${mins}m), no shard yet..."
    }
    Start-Sleep -Seconds $PollSec
  }
  if (-not $pfOk) {
    Write-Host "[$(Now)] PRE-FLIGHT FAILED (no shard). The arm64 path did not complete end-to-end -> NOT launching the fleet." -ForegroundColor Red
    Write-Host "  Diagnose: aws ec2 get-console-output --instance-id $pfId --region $REGION --output text"
    Write-Host "            aws s3 cp s3://$QBUCKET/$QPREFIX/results/h480/logs/cloud-init-output.log - --region $REGION"
    Write-Host "  Common causes: data not staged to S3, IAM 'quant-worker' missing, or an arm64 dep issue."
    $finalSt = Get-StateById $pfId
    if ($finalSt -eq "running" -or $finalSt -eq "pending") {
      aws ec2 terminate-instances --instance-ids $pfId --region $REGION 2>$null | Out-Null
      Write-Host "  terminated lingering pre-flight box $pfId."
    }
    try { Stop-Transcript | Out-Null } catch {}
    exit 2
  }
  Write-Host "[$(Now)] PRE-FLIGHT OK (shard produced). Purging h480 result + checkpoint so the fleet re-runs h=480 at FULL budget."
  Clear-H480Artifacts
} else {
  Write-Host "[$(Now)] -SkipPreflight set: launching the fleet WITHOUT the arm64 validation (your explicit opt-in)."
}

# ---- 4. launch the fleet babysitter (blocks until the fleet finishes) ------------------------------
Write-Host ""
Write-Host "[$(Now)] launching the vol-model fleet babysitter (-RunId $VolRunId)."
Write-Host "  (Open a second window and run scripts\aws\volmodel_watch.ps1 for read-only status.)"
Write-Host ""
& "$PSScriptRoot\volmodel_babysitter.ps1" -RunId $VolRunId
Write-Host ""
Write-Host "[$(Now)] fleet babysitter exited. queue_fleet_after_poker done -- check the watcher / per-horizon shards for status."
try { Stop-Transcript | Out-Null } catch {}
