# Distributed Deep Learning Inference Server 
**Production-Grade Async Architecture with Dynamic Batching** 

This project provides a production-grade distributed deep learning inference server built using Python async socket programming[cite: 48]. [cite_start]It enables thousands of concurrent clients to simultaneously transmit image data over TCP connections to a horizontally scalable inference cluster[cite: 49]. [cite_start]Images are processed through a pre-trained ResNet-50 convolutional neural network, returning top-5 class predictions in real time.

### 🚀 Key Features

* **Async Network I/O**: Uses `asyncio` and `uvloop` to handle 10,000+ concurrent connections on a single OS thread, eliminating the 8 MB/thread stack exhaustion seen in naive thread-per-client designs.
* **Dynamic Batching Engine**: Accumulates inference requests for up to 20 ms before dispatching a single batched GPU forward pass, increasing GPU utilization from 14% to 83%.
* **GIL-Bypass Preprocessing**: Offloads CPU-bound image decoding, resizing, and normalization to a 4-worker `ProcessPoolExecutor`.
* **Production Security Layer**: Includes SHA-256 API key authentication, per-client token-bucket rate limiting (10 req/s), bounded payload enforcement (20 MB cap), and magic-byte format validation.
* **Observability**: Emits structured JSON logs compatible with Grafana Loki and AWS CloudWatch, and exports Prometheus metrics (latency histograms, request counters, GPU utilization).
* **FastAPI REST Interface**: Thin HTTP layer providing `POST /predict`, `GET /health`, `GET /metrics`, and auto-generated OpenAPI documentation.
* **Graceful Shutdown**: SIGTERM-aware handler with configurable connection-draining to ensure zero-connection-drop rolling deployments in Kubernetes.

---

### 🏗️ System Architecture

The core architectural advancement is a three-layer async model:
1.  **Layer 1 — Async Network I/O**: `asyncio.start_server()` with a `uvloop` event loop policy handles all TCP connections via epoll.
2.  **Layer 2 — CPU Preprocessing Pool**: Image decode and manipulation operations are routed to worker processes via `loop.run_in_executor()`, safely avoiding the Python Global Interpreter Lock (GIL).
3.  **Layer 3 — GPU Batch Engine**: Preprocessed numpy tensors are accumulated into batches (up to 16 images) and dispatched to the GPU, serialized by an `asyncio.Semaphore`. Results are distributed back to waiting coroutines using `asyncio.Future`.

---

### 📊 Performance & Benchmarks

Benchmarks were conducted over a Docker bridge network using 500-image warm-started runs against an RTX 3060 GPU. 

* **Single-Worker (Batch=8)**: Achieves 42.1 req/s with a P95 latency of 195 ms.
* **Single-Worker (Batch=16)**: Achieves 61.8 req/s with a P95 latency of 210 ms—an 11x improvement over the 5.4 req/s naive single-threaded baseline.
* **Multi-Worker (3 Workers, Batch=16)**: Achieves 183.0 req/s with a P95 latency of 98 ms (81% GPU utilization).
* **Error Rate**: 0% error rate maintained across all configurations.

---

### 💻 Technology Stack

* ]**Core**: Python | asyncio | uvloop 
* **ML Framework**: PyTorch | torchvision (ResNet-50) 
* **API & Metrics**: FastAPI | Pydantic v2 | [cite_start]Prometheus 
* **Security & Data**: hashlib | JSON | [cite_start]PIL | numpy 

---

### 🛠️ Getting Started

#### Starting the Server
The FastAPI REST interface wraps the dynamic batch engine. To run the server:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
