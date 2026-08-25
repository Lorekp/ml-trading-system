# Rich READ-ONLY progress + ETA dashboard for the vol-model fleet (run in a 2nd window).
#
# The fleet babysitter only knows done/live/idle per horizon. This watcher adds real progress + ETA by
# reading signals the running boxes already produce (it NEVER launches/terminates anything):
#   * DONE      = the result shard exists in S3                  (authoritative)
#   * progress  = count of completed-month {YYYY-MM}.txt files mirrored to s3://.../checkpoints/h{H}/
#                 (the on-box CheckpointSyncer flushes every ~5 min; run_horizon_job.py CKPT_SYNC_SECONDS)
#   * total + per-month timing = parsed from the box's own console log lines
#                 "[N/M] 2025-XX: qlike=... (NNNs)"  (get-console-output)
#   * ETA       = (M - N) * recent-average month-seconds
#
# Run (your AWS creds), Ctrl-C to stop (does NOT touch any box):
#   powershell -ExecutionPolicy Bypass -File scripts\aws\volmodel_watch.ps1 -RunId volrun1
#
# ASCII ONLY (PS 5.1).

param(
  [string]$RunId    = "volrun1",
  [int[]]$Horizons  = @(5, 15, 30, 60, 240, 480),
  [int]$PollSec     = 90
)

$ErrorActionPreference = "Continue"
$REGION = if ($env:QUANT_REGION) { $env:QUANT_REGION } else { "eu-central-1" }
if ([string]::IsNullOrWhiteSpace($env:QUANT_BUCKET)) { throw "QUANT_BUCKET must be set" }
$BUCKET = $env:QUANT_BUCKET.Trim()
$PREFIX = if ($env:QUANT_PREFIX) { $env:QUANT_PREFIX } else { "volmodel/v1" }
# For the per-horizon Optuna-budget verdict shown next to DONE (calls horizon_optuna_report.py once).
$REPO   = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PY     = Join-Path $REPO ".venv\Scripts\python.exe"
$OPTRPT = Join-Path $PSScriptRoot "horizon_optuna_report.py"

# OptV caches the Optuna-budget verdict per DONE horizon; FleetM is the fleet-wide total-month
# estimate. ALL progress is from S3/EC2 -- NO get-console-output (it is empty/unreliable for a
# long-running training box, which is why M/ETA never populated before).
$OptV   = @{}
$FleetM = 0
$WatchBaseN = @{}   # per-horizon (months, time) at first sighting -> OBSERVED pace for ETA
$WatchBaseT = @{}   # (resume-robust: a relaunched box's uptime can't tell us the true per-month time)

function Now { return (Get-Date -Format "HH:mm:ss") }

function Test-Shard([int]$h) {
  aws s3 ls "s3://$BUCKET/$PREFIX/results/h$h/shard.json" --region $REGION 2>$null | Out-Null
  return ($LASTEXITCODE -eq 0)
}
function Get-Box([int]$h) {
  # returns "id|launchISO" if a running/pending box exists, "" if none, "ERR" on a failed AWS call.
  $r = (aws ec2 describe-instances --region $REGION `
        --filters "Name=tag:Run,Values=$RunId-h$h" "Name=instance-state-name,Values=running,pending" `
        --query "Reservations[].Instances[].[InstanceId,LaunchTime]" --output text 2>$null)
  if ($LASTEXITCODE -ne 0) { return "ERR" }
  if (-not $r) { return "" }
  return (($r -join " ") -replace "\s+", " ").Trim()
}
function Get-CheckpointMonths([int]$h) {
  # count of completed-month booster files mirrored to checkpoints/h{H}/ ; -1 on a failed AWS call.
  $r = (aws s3 ls "s3://$BUCKET/$PREFIX/checkpoints/h$h/" --region $REGION 2>$null)
  if ($LASTEXITCODE -ne 0) { return -1 }
  if (-not $r) { return 0 }
  return (@($r -split "`n" | Where-Object { $_ -match "\d{4}-\d{2}\.txt\s*$" }).Count)
}
function Fmt-Dur([double]$seconds) {
  if ($seconds -le 0) { return "?" }
  $t = [TimeSpan]::FromSeconds([double]$seconds)
  if ($t.TotalHours -ge 1) { return ("{0}h{1:00}m" -f [int][Math]::Floor($t.TotalHours), $t.Minutes) }
  if ($t.TotalMinutes -ge 1) { return ("{0}m" -f [int][Math]::Floor($t.TotalMinutes)) }
  return ("{0}s" -f [int]$t.TotalSeconds)
}
function Uptime-Min([string]$launchIso) {
  if ($launchIso -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
    try { return [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes) } catch { return -1 }
  }
  return -1
}

$poll = 0
while ($true) {
  $poll++
  Clear-Host
  Write-Host "=== volmodel fleet '$RunId'  $(Now)  (poll #$poll, every ${PollSec}s; Ctrl-C to stop) ==="
  Write-Host ("  bucket s3://$BUCKET/$PREFIX  region $REGION")
  Write-Host ""
  $nDone = 0
  $maxEta = 0.0
  $anyLive = $false
  # A FINISHED horizon checkpointed EVERY month, so its checkpoint count == its total months (M).
  # Take the max across done horizons as the fleet-wide M (all span ~the same 2022-2026 window) for
  # live-horizon % + ETA. Monotonic (never shrinks).
  foreach ($h in $Horizons) {
    if (Test-Shard $h) { $cm = Get-CheckpointMonths $h; if ($cm -gt $FleetM) { $FleetM = $cm } }
  }
  foreach ($h in $Horizons) {
    if (Test-Shard $h) {
      $nDone++
      if (-not $OptV.ContainsKey($h) -or -not $OptV[$h]) {
        $vl = (& $PY $OPTRPT --horizon $h 2>$null | Select-String "=>" | Select-Object -First 1)
        if ($vl) { $OptV[$h] = ($vl.ToString() -replace '.*=>\s*', '').Trim() }
      }
      $vtxt = if ($OptV.ContainsKey($h) -and $OptV[$h]) { $OptV[$h] } else { "(optuna report pending)" }
      Write-Host ("  h{0,-4} : DONE   |  optuna: {1}" -f $h, $vtxt) -ForegroundColor Green
      continue
    }
    $box = Get-Box $h
    if ($box -eq "ERR") { Write-Host ("  h{0,-4} : (AWS poll error - retrying)" -f $h) -ForegroundColor DarkYellow; continue }
    if (-not $box) { Write-Host ("  h{0,-4} : idle (no box; awaiting launch/relaunch)" -f $h) -ForegroundColor DarkGray; continue }
    $anyLive = $true
    $id = (@($box.Split(" ") | Where-Object { $_ -like "i-*" }))[0]
    $upMin = Uptime-Min $box
    $up = if ($upMin -ge 0) { Fmt-Dur ($upMin * 60) } else { "?" }
    $n = Get-CheckpointMonths $h     # completed months from S3 checkpoints (reliable; <=5min lag)
    $line = "  h{0,-4} : live {1} up {2}" -f $h, $id, $up
    if ($n -lt 0) {
      $line += "  |  (S3 poll error)"
    } elseif ($n -eq 0) {
      $line += "  |  booting / first month not finished yet"
    } else {
      if ($FleetM -gt 0) {
        $pct = [int](100.0 * $n / $FleetM); if ($pct -gt 100) { $pct = 100 }
        $line += "  |  month $n/~$FleetM ($pct%)"
      } else {
        $line += "  |  $n months trained"
      }
      # Per-month pace: PREFER the rate observed since this watcher started (Delta-months/Delta-time)
      # -- resume-robust. Fall back to box-uptime/months ONLY when that's plausible (>=60s/mo); a
      # resumed box (tiny uptime, many months) reads implausibly fast, so we say "measuring" instead
      # of printing an absurd ETA. The observed rate takes over once a month completes during watching.
      if (-not $WatchBaseN.ContainsKey($h)) { $WatchBaseN[$h] = $n; $WatchBaseT[$h] = Get-Date }
      $dMo = $n - $WatchBaseN[$h]
      $dSec = ((Get-Date) - $WatchBaseT[$h]).TotalSeconds
      $secPerMo = 0.0
      if ($dMo -gt 0 -and $dSec -gt 0) { $secPerMo = $dSec / $dMo }
      elseif (($upMin * 60.0 / $n) -ge 60) { $secPerMo = $upMin * 60.0 / $n }
      if ($FleetM -gt $n -and $secPerMo -gt 0) {
        $etaSec = ($FleetM - $n) * $secPerMo
        if ($etaSec -gt $maxEta) { $maxEta = $etaSec }
        $line += ("  |  ~{0:0.0}m/mo  ETA ~{1}" -f ($secPerMo / 60.0), (Fmt-Dur $etaSec))
      } else {
        $line += "  |  ETA measuring..."
      }
    }
    Write-Host $line
  }
  Write-Host ""
  $summary = "  fleet: $nDone/$($Horizons.Count) horizons DONE"
  if ($maxEta -gt 0) { $summary += "   |   slowest-horizon ETA ~$(Fmt-Dur $maxEta)  (h=5 is the long pole)" }
  Write-Host $summary -ForegroundColor Cyan
  if ($nDone -eq $Horizons.Count) { Write-Host "  ALL HORIZONS DONE."; break }
  if (-not $anyLive -and $nDone -lt $Horizons.Count) { Write-Host "  (no live boxes right now - babysitter will (re)launch the remaining horizons)" -ForegroundColor DarkGray }
  Start-Sleep -Seconds $PollSec
}
