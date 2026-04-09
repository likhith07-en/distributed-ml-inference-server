#!/usr/bin/env python3
"""
metrics.py — Prometheus Metrics for Async Inference Workers
Exposes /metrics endpoint (via FastAPI Instrumentator + custom gauges).
"""
import os, time, threading
from prometheus_client import Counter, Histogram, Gauge, start_http_server

METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
WORKER_ID    = os.getenv("WORKER_ID", "unknown")

# ── Metric definitions ────────────────────────────────────────────────────
inference_requests_total = Counter(
    "inference_requests_total",
    "Total inference requests processed",
    ["worker_id", "status"]           # status: success | error
)
inference_latency_ms = Histogram(
    "inference_latency_ms",
    "End-to-end inference latency (ms)",
    ["worker_id"],
    buckets=[10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 2000]
)
batch_size_histogram = Histogram(
    "inference_batch_size",
    "Number of images per GPU batch dispatch",
    ["worker_id"],
    buckets=[1, 2, 4, 8, 12, 16, 24, 32]
)
queue_depth_gauge = Gauge(
    "inference_queue_depth",
    "Current items waiting in the batch queue"
)
active_connections_gauge = Gauge(
    "active_client_connections",
    "Current number of active client connections",
    ["worker_id"]
)
gpu_utilization_gauge = Gauge(
    "gpu_utilization_percent",
    "GPU utilization (%)"
)
worker_health_gauge = Gauge(
    "worker_health_status",
    "1 = healthy, 0 = down",
    ["worker_id"]
)

class MetricsCollector:
    def __init__(self, worker_id: str = WORKER_ID):
        self.worker_id = worker_id
        worker_health_gauge.labels(worker_id=worker_id).set(1)

    def record(self, latency_ms: float, ok: bool, batch_size: int = 1):
        status = "success" if ok else "error"
        inference_requests_total.labels(worker_id=self.worker_id, status=status).inc()
        if ok:
            inference_latency_ms.labels(worker_id=self.worker_id).observe(latency_ms)
            batch_size_histogram.labels(worker_id=self.worker_id).observe(batch_size)

    def set_queue_depth(self, n: int):
        queue_depth_gauge.set(n)

def gpu_collector_thread():
    import subprocess
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode().strip()
            gpu_utilization_gauge.set(float(out.split("\n")[0]))
        except Exception:
            pass
        time.sleep(5)

def start_metrics_server():
    start_http_server(METRICS_PORT)
    t = threading.Thread(target=gpu_collector_thread, daemon=True)
    t.start()
    print(f"[Metrics] Prometheus /metrics on :{METRICS_PORT}")
