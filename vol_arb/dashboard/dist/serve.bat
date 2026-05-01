@echo off
REM Serve the pre-built BTC vol-arb backtest dashboard locally on Windows.
REM
REM Usage:
REM   serve.bat           - serves on http://127.0.0.1:8765
REM   serve.bat 8080      - serves on the chosen port
REM
REM Then open the URL in any modern browser.
REM
REM Requires: Python 3 (any 3.x) on PATH.

if "%1"=="" (set PORT=8765) else (set PORT=%1)
cd /d "%~dp0"
echo ==========================================================
echo   BTC vol-arb dashboard - static bundle
echo   Open: http://127.0.0.1:%PORT%
echo   Ctrl-C to stop
echo ==========================================================
python -m http.server %PORT% --bind 127.0.0.1
