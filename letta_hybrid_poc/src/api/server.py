"""
FastAPI server with SSE for real-time evaluation streaming.
"""
import asyncio
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .eval_runner import EvalRunner, EvalStatus


# FastAPI app
app = FastAPI(
    title="Letta Hybrid Evaluation API",
    description="Real-time evaluation streaming with SSE",
    version="1.0.0"
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global evaluation runner
eval_runner = EvalRunner()


class EvalRequest(BaseModel):
    """Evaluation request."""
    provider: str = "openai"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "mistral:latest"
    ollama_url: str = "http://localhost:11434"
    index_dir: str = "data/index"
    top_k: int = 5


class EvalResponse(BaseModel):
    """Evaluation response."""
    message: str
    status: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Letta Hybrid Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/eval/start": "Start evaluation",
            "GET /api/eval/stream": "Stream evaluation results (SSE)",
            "GET /api/eval/status": "Get evaluation status",
            "GET /api/eval/results": "Get completed results",
            "POST /api/eval/stop": "Stop evaluation"
        }
    }


@app.post("/api/eval/start", response_model=EvalResponse)
async def start_evaluation(request: EvalRequest):
    """
    Start evaluation in background.

    Returns immediately while evaluation runs.
    Use /api/eval/stream to watch progress.
    """
    try:
        if eval_runner.state.status == EvalStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Evaluation already running")

        # Start evaluation in background
        await eval_runner.start_evaluation(
            provider=request.provider,
            index_dir=Path(request.index_dir),
            openai_model=request.openai_model,
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
            top_k=request.top_k
        )

        return EvalResponse(
            message="Evaluation started",
            status="running"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eval/stream")
async def stream_evaluation():
    """
    Stream evaluation results via Server-Sent Events.

    Client should connect before starting evaluation.
    Events: evaluation_started, system_loading, system_loaded,
            query_started, query_completed, evaluation_completed, evaluation_error
    """
    async def event_generator():
        """Generate SSE events."""
        # Send initial status
        yield f"data: {json.dumps({'type': 'connected', 'status': eval_runner.state.status.value})}\n\n"

        # Queue for receiving events from eval_runner
        queue = asyncio.Queue()

        async def callback(event_type: str, data: dict):
            """Callback to receive events from eval_runner."""
            await queue.put((event_type, data))

        # Register callback
        eval_runner.add_callback(callback)

        try:
            # Stream events as they arrive
            while True:
                try:
                    # Wait for next event (with timeout for heartbeat)
                    event_type, data = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Send event
                    event_data = {
                        "type": event_type,
                        **data
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"

                    # Stop streaming after completion/error
                    if event_type in ["evaluation_completed", "evaluation_error", "evaluation_stopped"]:
                        break

                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield f": heartbeat\n\n"

        finally:
            # Remove callback when client disconnects
            if callback in eval_runner.callbacks:
                eval_runner.callbacks.remove(callback)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/eval/status")
async def get_status():
    """Get current evaluation status."""
    return eval_runner.get_state()


@app.get("/api/eval/results")
async def get_results():
    """Get all completed results."""
    results = eval_runner.get_results()
    state = eval_runner.get_state()

    return {
        "status": state["status"],
        "completed_queries": len(results),
        "total_queries": state["total_queries"],
        "results": results
    }


@app.post("/api/eval/stop", response_model=EvalResponse)
async def stop_evaluation():
    """Stop running evaluation."""
    if eval_runner.state.status != EvalStatus.RUNNING:
        raise HTTPException(status_code=400, detail="No evaluation running")

    eval_runner.stop_evaluation()

    return EvalResponse(
        message="Evaluation stop requested",
        status="stopping"
    )


@app.post("/api/eval/reset", response_model=EvalResponse)
async def reset_evaluation():
    """Reset evaluation state (clear results)."""
    if eval_runner.state.status == EvalStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot reset while evaluation is running")

    eval_runner.state = EvalStatus.IDLE
    eval_runner.state.results = []

    return EvalResponse(
        message="Evaluation state reset",
        status="idle"
    )

@app.get("/api/readme/header")
async def get_readme_header():
    """Get the first 3 lines of README.md."""
    try:
        readme_path = Path("README.md")
        if not readme_path.exists():
            raise HTTPException(status_code=404, detail="README.md not found")
        
        with open(readme_path, "r", encoding="utf-8") as f:
            lines = [next(f) for _ in range(3)]
        
        return {"header": lines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/readme/eval-query-tests")
async def get_readme_eval_query_tests():
    """Get the 'What Each Evaluation Query Tests' section from README.md."""
    try:
        readme_path = Path("README.md")
        if not readme_path.exists():
            raise HTTPException(status_code=404, detail="README.md not found")
        
        with open(readme_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        start_index = -1
        end_index = -1
        
        for i, line in enumerate(lines):
            if "What Each Evaluation Query Tests" in line:
                start_index = i
            if start_index != -1 and "### 2. Interactive Chat" in line:
                end_index = i
                break
        
        if start_index == -1:
            raise HTTPException(status_code=404, detail="Section not found")
            
        section = lines[start_index:end_index]
        
        return {"section": "".join(section)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
