# Regenerate report.json from the canonical BTC v2 trade log, then start Vite dev.
# Usage: .\run.ps1
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

python etl\build_report.py `
  --in   ..\output\BTC_v2_trade_log.parquet `
  --out  public\report.json

npm run dev
