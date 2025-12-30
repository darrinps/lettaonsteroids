@echo off
echo ================================================================================
echo Starting Letta Hybrid Evaluation API Server
echo ================================================================================
echo.
echo Server will start at: http://localhost:8000
echo.
echo API Documentation: http://localhost:8000/docs
echo Example UI: Open example_client.html in your browser
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python -m poetry run uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
