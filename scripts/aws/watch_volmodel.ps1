# At-a-glance READ-ONLY dashboard for the multi-horizon volatility-forecast cloud run.
#
# Ports scripts/watch_lbr6_search.ps1 from the earlier Pokerbot implementation,
# generalized to one row per HORIZON.
# Pairs with volmodel_babysitter.ps1 (which launches/terminates). This script NEVER launches or
# terminates anything -- it only polls S3 (result shards) + EC2 (live boxes by tag:Run).
#
# Run in a second window:
#   powershell -ExecutionPolicy Bypass -File scripts\aws\watch_volmodel.ps1 -RunId volrun1 -Horizons 5,15,30
#
# ASCII ONLY (PS 5.1 + BOM-less UTF-8 mangles non-ASCII).

param(
  [string]$RunId   = "volrun1",
  [int[]]$Horizons = @(5, 15, 30),
  [int]$Refresh    = 60
)
$REGION = if ($env:QUANT_REGION) { $env:QUANT_REGION } else { "eu-central-1" }
if ([string]::IsNullOrWhiteSpace($env:QUANT_BUCKET)) { throw "QUANT_BUCKET must be set" }
$BUCKET = $env:QUANT_BUCKET.Trim()
$PREFIX = if ($env:QUANT_PREFIX) { $env:QUANT_PREFIX } else { "volmodel/v1" }

function RunTag([int]$h) { return "$RunId-h$h" }
function ShardKey([int]$h) { return "$PREFIX/results/h$h/shard.json" }

$t0 = Get-Date

while ($true) {
  Clear-Host
  Write-Host ("=== volmodel forecast: run={0}   {1}   (watching {2}m) ===" -f `
    $RunId, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), [int]((Get-Date) - $t0).TotalMinutes)
  Write-Host ("bucket s3://{0}/{1}   region {2}" -f $BUCKET, $PREFIX, $REGION)
  Write-Host ""
  Write-Host ("  {0,-6} {1,-7} {2,-22} {3}" -f "horiz", "status", "box", "uptime")
  Write-Host ("  " + ("-" * 56))

  $nDone = 0; $nLive = 0; $nFail = 0
  foreach ($h in $Horizons) {
    # shard present -> done
    aws s3 ls "s3://$BUCKET/$(ShardKey $h)" --region $REGION 2>$null | Out-Null
    $done = $?

    # FAIL_* present -> note failure
    $failLs = aws s3 ls "s3://$BUCKET/$PREFIX/results/h$h/" --region $REGION 2>$null
    $hasFail = $false
    if ($failLs) { foreach ($ln in $failLs) { if ($ln -match "FAIL_") { $hasFail = $true } } }

    $tag = RunTag $h
    $boxRaw = aws ec2 describe-instances --region $REGION `
      --filters "Name=tag:Run,Values=$tag" "Name=instance-state-name,Values=running,pending" `
      --query "Reservations[].Instances[].[InstanceId,LaunchTime]" --output text 2>$null

    $boxId = "-"; $up = ""
    if ($boxRaw) {
      $f = ($boxRaw -split "\s+")
      $boxId = $f[0]
      if ($boxRaw -match "(\d{4}-\d{2}-\d{2}T[\d:.]+[+\-]\d{2}:?\d{2})") {
        # FIX-B (TZ): parse LaunchTime WITH offset so uptime is not inflated by the local offset.
        try {
          $m = [int](((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($Matches[1]).UtcDateTime).TotalMinutes)
          $up = "up ${m}m"
        } catch {}
      }
    }

    if ($done)            { $status = "DONE"; $nDone++ }
    elseif ($boxId -ne "-") { $status = "LIVE"; $nLive++ }
    elseif ($hasFail)     { $status = "FAIL"; $nFail++ }
    else                  { $status = "idle" }

    $color = switch ($status) { "DONE" { "Green" } "LIVE" { "Cyan" } "FAIL" { "Yellow" } default { "Gray" } }
    Write-Host ("  h{0,-5} {1,-7} {2,-22} {3}" -f $h, $status, $boxId, $up) -ForegroundColor $color
  }

  Write-Host ""
  Write-Host ("  {0}/{1} done   {2} live   {3} fail" -f $nDone, $Horizons.Count, $nLive, $nFail)

  if ($nDone -eq $Horizons.Count) {
    Write-Host ""
    Write-Host "ALL HORIZONS COMPLETE. Fetch each shard:" -ForegroundColor Green
    foreach ($h in $Horizons) {
      Write-Host ("  aws s3 cp s3://{0}/{1} -" -f $BUCKET, (ShardKey $h)) -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "Compare horizons on ABSOLUTE OOS QLIKE / OOS R^2 / RankIC (baseline_qlike is None off-15m)"
    Write-Host "AND downstream NC Sharpe / NC PnL -- per-YEAR stability, not whole-sample mean (anti-overfit)."
    break
  }
  if ($nFail -gt 0) {
    Write-Host ("  WARNING: {0} horizon(s) FAILED. Inspect: aws ec2 get-console-output --instance-id <id> --region {1}" -f $nFail, $REGION) -ForegroundColor Yellow
  }
  Write-Host ("(refresh {0}s; Ctrl-C to stop; read-only -- does not launch/terminate)" -f $Refresh)
  Start-Sleep -Seconds $Refresh
}
