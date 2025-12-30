"""
Example Python client for Letta Hybrid Evaluation API.

Usage:
    python example_client.py
"""
import requests
import json
import sseclient


API_BASE = "http://localhost:8000"


def start_evaluation():
    """Start evaluation."""
    print("Starting evaluation...")
    response = requests.post(f"{API_BASE}/api/eval/start", json={
        "provider": "openai",
        "openai_model": "gpt-4o-mini",
        "top_k": 5
    })
    print(f"Response: {response.json()}")
    return response.json()


def stream_results():
    """Stream evaluation results via SSE."""
    print("\nStreaming results...\n")

    headers = {'Accept': 'text/event-stream'}
    response = requests.get(f"{API_BASE}/api/eval/stream", headers=headers, stream=True)

    client = sseclient.SSEClient(response)

    for event in client.events():
        if event.data:
            data = json.loads(event.data)
            event_type = data.get('type')

            if event_type == 'connected':
                print(f"[CONNECTED] Status: {data.get('status')}")

            elif event_type == 'evaluation_started':
                print(f"[STARTED] Running {data.get('total_queries')} queries with {data.get('provider')} ({data.get('model')})")

            elif event_type == 'system_loading':
                print("[LOADING] Loading system components...")

            elif event_type == 'system_loaded':
                print("[LOADED] System ready\n")

            elif event_type == 'query_started':
                print(f"[QUERY {data.get('index')}/{data.get('total')}] {data.get('query')}")

            elif event_type == 'query_completed':
                baseline = data.get('baseline', {})
                augmented = data.get('augmented', {})
                print(f"  Baseline:  {baseline.get('latency_ms', 0):.0f}ms, Recall: {baseline.get('recall_at_k', 0):.2f}")
                print(f"  Augmented: {augmented.get('latency_ms', 0):.0f}ms, Recall: {augmented.get('recall_at_k', 0):.2f}\n")

            elif event_type == 'evaluation_completed':
                metrics = data.get('aggregate_metrics', {})
                baseline_metrics = metrics.get('baseline', {})
                augmented_metrics = metrics.get('augmented', {})

                print("\n" + "="*60)
                print("EVALUATION COMPLETE")
                print("="*60)
                print(f"Total queries: {data.get('total_queries')}")
                print(f"Duration: {data.get('duration_seconds', 0):.1f}s")
                print(f"\nBaseline:  {baseline_metrics.get('avg_latency_ms', 0):.0f}ms avg, {baseline_metrics.get('avg_recall_at_k', 0):.3f} recall")
                print(f"Augmented: {augmented_metrics.get('avg_latency_ms', 0):.0f}ms avg, {augmented_metrics.get('avg_recall_at_k', 0):.3f} recall")
                break

            elif event_type == 'evaluation_error':
                print(f"[ERROR] {data.get('error')}")
                break

            elif event_type == 'evaluation_stopped':
                print(f"[STOPPED] Evaluation stopped after {data.get('completed_queries')} queries")
                break


def get_status():
    """Get evaluation status."""
    response = requests.get(f"{API_BASE}/api/eval/status")
    return response.json()


def get_results():
    """Get all results."""
    response = requests.get(f"{API_BASE}/api/eval/results")
    return response.json()


def stop_evaluation():
    """Stop running evaluation."""
    print("\nStopping evaluation...")
    response = requests.post(f"{API_BASE}/api/eval/stop")
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    import sys

    print("="*60)
    print("Letta Hybrid Evaluation API - Example Client")
    print("="*60)

    # Check if server is running
    try:
        response = requests.get(API_BASE)
        print(f"Connected to: {response.json()['name']}\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to API server at {API_BASE}")
        print("Start server with: poetry run python -m src.api.server")
        sys.exit(1)

    # Start and stream
    start_evaluation()
    stream_results()

    # Get final results
    print("\nFetching final results...")
    results = get_results()
    print(f"Total results: {results['completed_queries']}")
