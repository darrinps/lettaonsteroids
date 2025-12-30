@echo off
echo ================================================================================
echo Letta Hybrid POC - Test Suite
echo ================================================================================
echo.

echo [1/3] Testing module imports...
echo.
poetry run python test_imports.py
if errorlevel 1 (
    echo.
    echo [FAIL] Import test failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.

echo [2/3] Testing OpenAI connection...
echo.
poetry run python test_openai.py
if errorlevel 1 (
    echo.
    echo [FAIL] OpenAI test failed!
    echo Check that OPENAI_API_KEY is set in .env file
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.

echo [3/3] Checking index...
echo.
poetry run python -m src.cli.ingest info
if errorlevel 1 (
    echo.
    echo [FAIL] Index not found!
    echo Run: poetry run python -m src.cli.ingest build
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.
echo [SUCCESS] All tests passed!
echo ================================================================================
echo.
echo You can now run:
echo   poetry run python -m src.cli.chat run --provider openai
echo.
pause
