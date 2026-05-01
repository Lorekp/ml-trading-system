@echo off
REM Rebuild BTC vol-arb dashboard from the latest trade-log parquet.
REM Regenerates public/report.json and launches the Vite dev server.

setlocal
set DASH=C:\Claude\ml-trading-system\vol_arb\dashboard
set PARQUET=C:\Claude\ml-trading-system\vol_arb\output\BTC_v2_trade_log.parquet

if not exist "%PARQUET%" (
    echo [ERROR] Parquet not found: %PARQUET%
    pause
    exit /b 1
)

cd /d "%DASH%"

echo.
echo [1/2] Rebuilding report.json from parquet...
echo.
python etl\build_report.py --in "%PARQUET%" --out public\report.json
if errorlevel 1 (
    echo.
    echo [ERROR] ETL failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting dev server at http://localhost:5173 ...
echo      (Ctrl+C in this window to stop)
echo.

REM Open browser after a short delay, then block on the dev server.
start "" cmd /c "timeout /t 3 >nul && start http://localhost:5173"
call npm run dev

endlocal
