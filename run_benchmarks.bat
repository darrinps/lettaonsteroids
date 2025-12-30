@echo off
REM Letta vs Mem0 Benchmark Runner
REM Usage: run_benchmarks.bat [backend] [sessions] [noise]
REM   backend: letta, mem0, or both (default: both)
REM   sessions: number of test sessions (default: 1)
REM   noise: noise ratio 0.0-1.0 (default: 0.0)

setlocal enabledelayedexpansion

REM Parse arguments with defaults
set "BACKEND=%~1"
set "SESSIONS=%~2"
set "NOISE=%~3"

if "%BACKEND%"=="" set "BACKEND=both"
if "%SESSIONS%"=="" set "SESSIONS=1"
if "%NOISE%"=="" set "NOISE=0.0"

echo ======================================
echo   Letta vs Mem0 Benchmark Runner
echo ======================================
echo Backend:  %BACKEND%
echo Sessions: %SESSIONS%
echo Noise:    %NOISE%
echo ======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    exit /b 1
)

REM Create and activate virtual environment
if not exist .venv (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

REM Upgrade pip and install dependencies
echo [INFO] Installing dependencies...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing...
)

python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    exit /b 1
)

REM Install openai package (needed for embeddings)
python -c "import openai" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing openai package...
    python -m pip install openai --quiet
    if errorlevel 1 (
        echo [WARNING] Failed to install openai package. Similarity metrics will be disabled.
    )
)

REM Create output directory
if not exist out mkdir out

REM Check if required environment variables are set
if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] OPENAI_API_KEY not set. Similarity metrics will be disabled.
)

REM Check if backend servers are running (optional warning)
if "%BACKEND%"=="letta" goto check_letta
if "%BACKEND%"=="both" goto check_letta
goto check_mem0

:check_letta
set "LETTA_URL=%LETTA_BASE_URL%"
if "%LETTA_URL%"=="" set "LETTA_URL=http://localhost:8283"
curl -s --connect-timeout 2 "%LETTA_URL%" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Cannot reach Letta server at %LETTA_URL%
    echo [WARNING] Make sure Letta server is running before testing Letta backend.
)
if "%BACKEND%"=="letta" goto run_benchmark

:check_mem0
if "%BACKEND%"=="mem0" goto check_mem0_server
if "%BACKEND%"=="both" goto check_mem0_server
goto run_benchmark

:check_mem0_server
set "MEM0_URL=%MEM0_BASE_URL%"
if "%MEM0_URL%"=="" set "MEM0_URL=http://localhost:3000"
curl -s --connect-timeout 2 "%MEM0_URL%" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Cannot reach Mem0 server at %MEM0_URL%
    echo [WARNING] Make sure Mem0 server is running before testing Mem0 backend.
)

:run_benchmark
REM Run benchmark
echo.
echo [INFO] Running benchmarks...
echo.

python -m src.benchmark --backend %BACKEND% --sessions %SESSIONS% --noise %NOISE% --out out\results.json --csv out\results.csv

if errorlevel 1 (
    echo.
    echo [ERROR] Benchmark failed!
    exit /b 1
)

REM Show results
echo.
echo ======================================
echo   Benchmark Complete!
echo ======================================
echo Results saved to:
echo   - out\results.json
echo   - out\results.csv
echo.
echo To view results:
echo   type out\results.csv
echo.

endlocal
