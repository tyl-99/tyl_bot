@echo off
echo Starting Trader Dashboard Application from root...
echo.

REM Change to web_app directory
cd web_app
if errorlevel 1 (
    echo ERROR: Failed to change to web_app directory. Make sure you are running this from the MyTrader root.
    pause
    exit /b 1
)

echo Step 1: Processing data... (Minimal output)
python data_processor.py
if errorlevel 1 (
    echo ERROR: Failed to process data. Check web_app/data_processor.py output for details.
    pause
    exit /b 1
)

echo.
echo Data processed successfully!
echo.
echo Step 2: Starting Next.js development server in a new window...
echo Dashboard will be available at: http://localhost:3000
echo.

REM Start Next.js development server in a new window
start "Next.js Dev Server" cmd /c npm run dev

echo Waiting for the server to become available on http://localhost:3000 ...
set "URL=http://localhost:3000"
set "MAX_TRIES=60"
set "SLEEP_SECONDS=2"

for /l %%i in (1,1,%MAX_TRIES%) do (
    curl -s %URL% >nul 2>&1
    if not errorlevel 1 goto launch_browser
    timeout /t %SLEEP_SECONDS% >nul
)

echo ERROR: Timed out waiting for %URL%
goto end

:launch_browser
echo Launching Chrome...
where chrome >nul 2>nul
if %ERRORLEVEL%==0 (
    start "" chrome %URL%
) else (
    echo Chrome not found in PATH. Opening with default browser...
    start "" %URL%
)

echo.
echo Next.js is running in a separate window. You can close this window.

:end
