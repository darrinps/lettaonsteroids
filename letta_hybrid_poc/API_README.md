# Letta Hybrid Evaluation API

Real-time evaluation streaming API with Server-Sent Events (SSE) for building UIs.

---

## 🎯 Who Is This For?

**UI/Frontend Developers:**
- ✅ You just need HTTP and EventSource (built into browsers)
- ✅ No Python, Poetry, or backend dependencies required
- ✅ Open `example_client.html` and start building
- 👉 Skip to [Usage Examples](#usage-examples)

**Backend Developers:**
- ✅ You need Python 3.11+, Poetry, and OpenAI API Key
- ✅ See main [README.md](README.md) for backend setup
- 👉 Then come back here for API documentation

---

## Features

- ✅ **Real-time streaming** - Results appear as each query completes
- ✅ **Start/Stop control** - Full control over evaluation lifecycle
- ✅ **REST + SSE** - Standard web technologies
- ✅ **CORS enabled** - Works from any origin (configurable)
- ✅ **Auto-documented** - FastAPI generates OpenAPI/Swagger docs

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Start Server

**Windows:**
```bash
start_server.bat
```

**Or manually:**
```bash
poetry run uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: http://localhost:8000

### 3. View API Documentation

Open in browser: http://localhost:8000/docs

Interactive Swagger UI with all endpoints documented.

### 4. Try Example Clients

**Web UI (Recommended):**
1. Open `example_client.html` in your browser
2. Click "Start Evaluation"
3. Watch real-time results

**Python CLI:**
```bash
poetry run python example_client.py
```

## API Endpoints

### POST /api/eval/start

Start evaluation in background.

**Request:**
```json
{
  "provider": "openai",
  "openai_model": "gpt-4o-mini",
  "ollama_model": "mistral:latest",
  "ollama_url": "http://localhost:11434",
  "index_dir": "data/index",
  "top_k": 5
}
```

**Response:**
```json
{
  "message": "Evaluation started",
  "status": "running"
}
```

### GET /api/eval/stream

Stream evaluation results via Server-Sent Events.

**Event Types:**
- `connected` - Initial connection
- `evaluation_started` - Evaluation begins
- `system_loading` - Loading components
- `system_loaded` - System ready
- `query_started` - Query begins
- `query_completed` - Query finished with results
- `evaluation_completed` - All queries done
- `evaluation_error` - Error occurred
- `evaluation_stopped` - User stopped evaluation

**Example Event:**
```json
{
  "type": "query_completed",
  "index": 1,
  "total": 8,
  "query": "What is Cedar?",
  "baseline": {
    "latency_ms": 2800,
    "recall_at_k": 1.0,
    "answer": "..."
  },
  "augmented": {
    "latency_ms": 3500,
    "recall_at_k": 1.0,
    "answer": "..."
  }
}
```

### GET /api/eval/status

Get current evaluation status.

**Response:**
```json
{
  "status": "running",
  "current_query": 3,
  "total_queries": 8,
  "completed_queries": 2,
  "started_at": 1234567890.123
}
```

### GET /api/eval/results

Get all completed results.

**Response:**
```json
{
  "status": "running",
  "completed_queries": 3,
  "total_queries": 8,
  "results": [...]
}
```

### POST /api/eval/stop

Stop running evaluation.

**Response:**
```json
{
  "message": "Evaluation stop requested",
  "status": "stopping"
}
```

### POST /api/eval/reset

Reset evaluation state (clears results).

**Response:**
```json
{
  "message": "Evaluation state reset",
  "status": "idle"
}
```

## Usage Examples

### JavaScript/Browser

```javascript
// Start evaluation
const response = await fetch('http://localhost:8000/api/eval/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        provider: 'openai',
        openai_model: 'gpt-4o-mini',
        top_k: 5
    })
});

// Stream results
const eventSource = new EventSource('http://localhost:8000/api/eval/stream');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'query_completed') {
        console.log(`Query ${data.index}: ${data.query}`);
        console.log(`Baseline: ${data.baseline.latency_ms}ms`);
        console.log(`Augmented: ${data.augmented.latency_ms}ms`);
    }

    if (data.type === 'evaluation_completed') {
        console.log('Done!', data.aggregate_metrics);
        eventSource.close();
    }
};

// Stop evaluation
await fetch('http://localhost:8000/api/eval/stop', { method: 'POST' });
```

### Python

```python
import requests
import sseclient

# Start evaluation
requests.post('http://localhost:8000/api/eval/start', json={
    'provider': 'openai',
    'openai_model': 'gpt-4o-mini',
    'top_k': 5
})

# Stream results
response = requests.get('http://localhost:8000/api/eval/stream',
                       headers={'Accept': 'text/event-stream'},
                       stream=True)

client = sseclient.SSEClient(response)
for event in client.events():
    data = json.loads(event.data)

    if data['type'] == 'query_completed':
        print(f"Query {data['index']}: {data['query']}")
        print(f"Baseline: {data['baseline']['latency_ms']}ms")

    if data['type'] == 'evaluation_completed':
        print('Done!', data['aggregate_metrics'])
        break
```

### React Example

```jsx
import { useState, useEffect } from 'react';

function Evaluation() {
    const [status, setStatus] = useState('idle');
    const [results, setResults] = useState([]);
    const [metrics, setMetrics] = useState(null);

    const startEvaluation = async () => {
        await fetch('http://localhost:8000/api/eval/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: 'openai',
                openai_model: 'gpt-4o-mini'
            })
        });

        const eventSource = new EventSource('http://localhost:8000/api/eval/stream');

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'query_completed') {
                setResults(prev => [...prev, data]);
            }

            if (data.type === 'evaluation_completed') {
                setMetrics(data.aggregate_metrics);
                setStatus('completed');
                eventSource.close();
            }
        };
    };

    return (
        <div>
            <button onClick={startEvaluation}>Start</button>
            <div>Status: {status}</div>
            <div>Completed: {results.length}</div>
            {metrics && <div>Avg Latency: {metrics.augmented.avg_latency_ms}ms</div>}
        </div>
    );
}
```

## Architecture

```
┌─────────────────┐
│   Web UI        │ (React/Vue/Svelte/HTML)
│   Mobile App    │ (React Native/Flutter)
│   Desktop App   │ (Electron/Tauri)
└────────┬────────┘
         │
         │ HTTP REST + SSE
         │
┌────────▼────────┐
│  FastAPI Server │ (Port 8000)
│   - REST API    │
│   - SSE Stream  │
│   - CORS        │
└────────┬────────┘
         │
         │ Async
         │
┌────────▼────────┐
│  Eval Runner    │
│   - Background  │
│   - Streaming   │
│   - Callbacks   │
└────────┬────────┘
         │
         │
┌────────▼────────┐
│  Letta System   │
│   - Baseline    │
│   - Augmented   │
│   - Metrics     │
└─────────────────┘
```

## Development

### Run with Auto-reload

```bash
poetry run uvicorn src.api.server:app --reload
```

Code changes will automatically restart the server.

### Check Server Health

```bash
curl http://localhost:8000
```

### View Logs

Server logs appear in terminal with request/response details.

## Production Deployment

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export CORS_ORIGINS=https://yourdomain.com
```

### Run with Gunicorn

```bash
gunicorn src.api.server:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker (Example)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install poetry && poetry install
CMD ["poetry", "run", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000
```

## Troubleshooting

### CORS Errors

Server has CORS enabled for all origins by default. To restrict:

```python
# In src/api/server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    ...
)
```

### SSE Connection Drops

- SSE has 30s heartbeat to keep connection alive
- If connection drops, client should reconnect
- Use `EventSource` built-in reconnection (browser handles this)

### Port Already in Use

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:8000 | xargs kill
```

## License

MIT
