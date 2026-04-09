#!/usr/bin/env python3
"""
grpc_server.py — Production gRPC Inference Server (alternative transport)

Implements inference.proto InferenceService:
  • Predict          — unary RPC
  • PredictStream    — bidirectional streaming RPC
  • HealthCheck      — unary health probe

Wires with:
  model_loader.py   → ModelLoader (FP16/INT8 ResNet-50, preprocessing)
  batch_engine.py   → DynamicBatchEngine (≤16 imgs / 20 ms batching)
  security.py       → validate_image_bytes, VALID_KEY_HASHES, rate_limiter
  logging_config.py → setup_logging (structured JSON)
  metrics.py        → MetricsCollector (Prometheus counters/histograms)

Generated stubs required (run once):
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

Run:
  WORKER_PORT=50051 WORKER_ID=grpc-w1 python grpc_server.py
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import AsyncIterator

import grpc
import grpc.aio

import inference_pb2
import inference_pb2_grpc

from batch_engine import DynamicBatchEngine
from logging_config import setup_logging
from metrics import MetricsCollector
from model_loader import ModelLoader
from security import VALID_KEY_HASHES, rate_limiter, validate_image_bytes

# ── Configuration ────────────────────────────────────────────────────────────
GRPC_PORT: int = int(os.getenv("WORKER_PORT", "50051"))
WORKER_ID: str = os.getenv("WORKER_ID", "grpc-worker-default")
MAX_PAYLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB hard cap
PREPROCESS_WORKERS: int = 4               # ProcessPoolExecutor pool size

log = logging.getLogger("grpc_server")

# ── Globals (initialised in serve()) ────────────────────────────────────────
_model: ModelLoader | None = None
_engine: DynamicBatchEngine | None = None
_cpu_pool: ProcessPoolExecutor | None = None
_metrics: MetricsCollector | None = None


# ── Auth helper ──────────────────────────────────────────────────────────────

def _authenticate(context: grpc.aio.ServicerContext) -> str | None:
    """
    Validate 'x-api-key' metadata against SHA-256 hashes.

    Returns the client_id string on success, or None if key is missing/invalid.
    Sends the appropriate gRPC status code on failure.
    """
    metadata = dict(context.invocation_metadata())
    raw_key = metadata.get("x-api-key", "")

    if not raw_key:
        return None

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return VALID_KEY_HASHES.get(key_hash)


async def _check_auth(
    context: grpc.aio.ServicerContext,
) -> str | None:
    """Abort with UNAUTHENTICATED if key missing; PERMISSION_DENIED if invalid."""
    metadata = dict(context.invocation_metadata())
    raw_key = metadata.get("x-api-key", "")

    if not raw_key:
        await context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            "Missing x-api-key metadata.",
        )
        return None

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    client_id = VALID_KEY_HASHES.get(key_hash)

    if client_id is None:
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "Invalid API key.",
        )
        return None

    if not rate_limiter.allow(client_id):
        await context.abort(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "Rate limit exceeded (10 req/s per client).",
        )
        return None

    return client_id


def _build_predictions(
    raw: list[dict],
) -> list[inference_pb2.Prediction]:
    """Convert DynamicBatchEngine output to protobuf Prediction messages."""
    return [
        inference_pb2.Prediction(
            rank=p["rank"],
            label=p["label"],
            confidence=float(p["confidence"]),
        )
        for p in raw
    ]


# ── Servicer ─────────────────────────────────────────────────────────────────

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    Full implementation of InferenceService (inference.proto).

    All three RPCs are async-native (grpc.aio).
    GPU inference is serialised through DynamicBatchEngine; CPU
    preprocessing runs in a separate ProcessPoolExecutor to bypass the GIL.
    """

    # ── Unary Predict ────────────────────────────────────────────────────────

    async def Predict(
        self,
        request: inference_pb2.PredictRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PredictResponse:
        """
        Unary inference RPC.

        1. Auth + rate limit (x-api-key metadata).
        2. Magic-byte image validation.
        3. CPU preprocessing in ProcessPoolExecutor.
        4. Dynamic batch engine submit → GPU forward pass.
        5. Return PredictResponse with top-K predictions.
        """
        assert _model is not None and _engine is not None and _cpu_pool is not None

        request_id = request.request_id or str(uuid.uuid4())
        t0 = time.perf_counter()

        client_id = await _check_auth(context)
        if client_id is None:
            return inference_pb2.PredictResponse(
                request_id=request_id,
                error="Authentication failed.",
            )

        image_bytes = bytes(request.image_data)

        # Payload size guard
        if len(image_bytes) > MAX_PAYLOAD_BYTES or len(image_bytes) < 100:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Payload size {len(image_bytes)} bytes out of allowed range.",
            )
            return inference_pb2.PredictResponse(
                request_id=request_id,
                error="Invalid payload size.",
            )

        # Magic-byte validation
        try:
            validate_image_bytes(image_bytes)
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return inference_pb2.PredictResponse(
                request_id=request_id,
                error=str(e),
            )

        # CPU preprocessing (GIL-free via ProcessPoolExecutor)
        loop = asyncio.get_running_loop()
        try:
            img_np = await loop.run_in_executor(
                _cpu_pool,
                _model.preprocess_to_numpy,
                image_bytes,
            )
        except Exception as exc:
            log.error("Preprocessing failed", extra={"request_id": request_id, "error": str(exc)})
            await context.abort(grpc.StatusCode.INTERNAL, f"Preprocessing error: {exc}")
            return inference_pb2.PredictResponse(
                request_id=request_id,
                error=str(exc),
            )

        # Dynamic batching → GPU inference
        try:
            preds_raw = await _engine.submit(img_np)
        except Exception as exc:
            log.error("Inference failed", extra={"request_id": request_id, "error": str(exc)})
            if _metrics:
                _metrics.record(latency_ms=0, ok=False)
            await context.abort(grpc.StatusCode.INTERNAL, f"Inference error: {exc}")
            return inference_pb2.PredictResponse(
                request_id=request_id,
                error=str(exc),
            )

        ms = (time.perf_counter() - t0) * 1000
        if _metrics:
            _metrics.record(latency_ms=ms, ok=True)

        log.info(
            "predict",
            extra={
                "request_id": request_id,
                "client": client_id,
                "latency_ms": round(ms, 1),
                "top": preds_raw[0]["label"] if preds_raw else "—",
                "transport": "grpc-unary",
            },
        )

        # Honour optional top_k override (default 5)
        top_k = request.top_k if request.top_k > 0 else 5
        return inference_pb2.PredictResponse(
            request_id=request_id,
            predictions=_build_predictions(preds_raw[:top_k]),
            inference_time_ms=round(ms, 1),
            worker_id=WORKER_ID,
        )

    # ── Bidirectional Streaming PredictStream ────────────────────────────────

    async def PredictStream(
        self,
        request_iterator: AsyncIterator[inference_pb2.PredictRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[inference_pb2.PredictResponse]:
        """
        Bidirectional streaming RPC.

        Auth is checked once per stream (on the first message metadata).
        Each incoming PredictRequest is processed independently through the
        DynamicBatchEngine — multiple in-flight requests naturally form
        batches just as in the TCP async_worker.

        Stream terminates when the client closes the request side.
        """
        assert _model is not None and _engine is not None and _cpu_pool is not None

        # Auth on stream open (metadata arrives with the initial call)
        client_id = await _check_auth(context)
        if client_id is None:
            return

        loop = asyncio.get_running_loop()

        async for request in request_iterator:
            request_id = request.request_id or str(uuid.uuid4())
            t0 = time.perf_counter()

            image_bytes = bytes(request.image_data)

            # Validate size & magic bytes
            try:
                if len(image_bytes) > MAX_PAYLOAD_BYTES or len(image_bytes) < 100:
                    raise ValueError(f"Payload size {len(image_bytes)} bytes out of range.")
                validate_image_bytes(image_bytes)
            except ValueError as exc:
                # Do not abort the stream — yield an error response and continue
                yield inference_pb2.PredictResponse(
                    request_id=request_id,
                    error=str(exc),
                    worker_id=WORKER_ID,
                )
                continue

            # CPU preprocessing
            try:
                img_np = await loop.run_in_executor(
                    _cpu_pool,
                    _model.preprocess_to_numpy,
                    image_bytes,
                )
            except Exception as exc:
                log.error(
                    "Stream preprocessing failed",
                    extra={"request_id": request_id, "error": str(exc)},
                )
                yield inference_pb2.PredictResponse(
                    request_id=request_id,
                    error=f"Preprocessing error: {exc}",
                    worker_id=WORKER_ID,
                )
                continue

            # Batch engine → GPU
            try:
                preds_raw = await _engine.submit(img_np)
            except Exception as exc:
                log.error(
                    "Stream inference failed",
                    extra={"request_id": request_id, "error": str(exc)},
                )
                if _metrics:
                    _metrics.record(latency_ms=0, ok=False)
                yield inference_pb2.PredictResponse(
                    request_id=request_id,
                    error=f"Inference error: {exc}",
                    worker_id=WORKER_ID,
                )
                continue

            ms = (time.perf_counter() - t0) * 1000
            if _metrics:
                _metrics.record(latency_ms=ms, ok=True)

            top_k = request.top_k if request.top_k > 0 else 5
            log.info(
                "predict_stream",
                extra={
                    "request_id": request_id,
                    "client": client_id,
                    "latency_ms": round(ms, 1),
                    "top": preds_raw[0]["label"] if preds_raw else "—",
                    "transport": "grpc-stream",
                },
            )

            yield inference_pb2.PredictResponse(
                request_id=request_id,
                predictions=_build_predictions(preds_raw[:top_k]),
                inference_time_ms=round(ms, 1),
                worker_id=WORKER_ID,
            )

    # ── HealthCheck ──────────────────────────────────────────────────────────

    async def HealthCheck(
        self,
        request: inference_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.HealthCheckResponse:
        """
        Lightweight health probe — no auth required.

        Returns worker_id, device, queue depth, and a rough P95 latency
        estimate (sampled from the last 100 requests via MetricsCollector
        if available, otherwise 0.0).
        """
        assert _model is not None and _engine is not None

        queue_depth: int = _engine.queue.qsize()
        device_str: str = str(_model.device)

        # P95 latency: read from Prometheus histogram if available
        p95: float = 0.0
        try:
            from prometheus_client import REGISTRY
            for metric in REGISTRY.collect():
                if metric.name == "inference_latency_ms":
                    for sample in metric.samples:
                        if sample.name.endswith("_bucket") and sample.labels.get("quantile") == "0.95":
                            p95 = float(sample.value)
                            break
        except Exception:
            pass

        return inference_pb2.HealthCheckResponse(
            healthy=True,
            worker_id=WORKER_ID,
            device=device_str,
            queue_depth=queue_depth,
            p95_latency_ms=p95,
        )


# ── Server lifecycle ─────────────────────────────────────────────────────────

async def serve() -> None:
    global _model, _engine, _cpu_pool, _metrics

    setup_logging()

    log.info("Initialising ModelLoader …")
    _model = ModelLoader()
    _model.warmup()

    _cpu_pool = ProcessPoolExecutor(max_workers=PREPROCESS_WORKERS)
    _engine = DynamicBatchEngine(_model)
    await _engine.start()

    _metrics = MetricsCollector(worker_id=WORKER_ID)

    # ── gRPC server ─────────────────────────────────────────────────────────
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", MAX_PAYLOAD_BYTES + 1024),
            ("grpc.max_send_message_length",    MAX_PAYLOAD_BYTES + 1024),
            # Keep-alive: send a ping every 30 s; tolerate 5 missed pings
            ("grpc.keepalive_time_ms",               30_000),
            ("grpc.keepalive_timeout_ms",            10_000),
            ("grpc.keepalive_permit_without_calls",       1),
            ("grpc.http2.max_pings_without_data",         0),
            ("grpc.http2.min_time_between_pings_ms", 10_000),
        ]
    )

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), server
    )

    listen_addr = f"0.0.0.0:{GRPC_PORT}"
    server.add_insecure_port(listen_addr)

    await server.start()
    log.info(
        "gRPC server ready",
        extra={"addr": listen_addr, "worker_id": WORKER_ID, "device": str(_model.device)},
    )

    # ── Graceful shutdown on SIGTERM / SIGINT ────────────────────────────────
    import signal

    stop_event = asyncio.Event()

    def _on_signal() -> None:
        log.info("Shutdown signal — draining gRPC server (30 s) …")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    await stop_event.wait()
    await server.stop(grace=30)
    _cpu_pool.shutdown(wait=False)
    log.info("gRPC server stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(serve())
