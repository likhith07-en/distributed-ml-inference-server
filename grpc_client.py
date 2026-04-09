#!/usr/bin/env python3
"""
grpc_client.py — gRPC Inference Client

Matches grpc_server.py / inference.proto InferenceService.

Modes:
  predict       — single unary RPC   (default)
  stream        — bidirectional streaming over a directory of images
  health        — HealthCheck RPC

Usage:
  # Unary prediction
  python grpc_client.py predict dog.jpg

  # Unary with custom host/port/top-k
  python grpc_client.py predict dog.jpg --host localhost --port 50051 --top-k 3

  # Streaming: send every JPEG/PNG in a directory
  python grpc_client.py stream ./test_images/

  # Health check
  python grpc_client.py health

Environment:
  GRPC_API_KEY  — override the default dev key (default: dev-key-abc123)
  GRPC_HOST     — server host (default: localhost)
  GRPC_PORT     — server port (default: 50051)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import grpc
import grpc.aio

import inference_pb2
import inference_pb2_grpc

# ── Defaults (overridable via env or CLI) ────────────────────────────────────
DEFAULT_API_KEY: str = os.getenv("GRPC_API_KEY", "dev-key-abc123")
DEFAULT_HOST: str    = os.getenv("GRPC_HOST",    "localhost")
DEFAULT_PORT: int    = int(os.getenv("GRPC_PORT", "50051"))

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _print_predictions(
    response: inference_pb2.PredictResponse,
    *,
    image_name: str = "",
    elapsed_ms: float | None = None,
) -> None:
    """Pretty-print a PredictResponse to stdout."""
    timing = f" [{elapsed_ms:.1f} ms client-side]" if elapsed_ms is not None else ""
    src = f" for '{image_name}'" if image_name else ""
    print(f"\nTop Predictions{src}{timing}  [worker: {response.worker_id}]")
    print(f"  request_id      : {response.request_id}")
    print(f"  inference_time  : {response.inference_time_ms:.1f} ms")

    if response.error:
        print(f"  ⚠  ERROR: {response.error}")
        return

    for p in response.predictions:
        bar = "█" * int(p.confidence / 2)
        print(f"  {p.rank:>2}. {p.label:<35} {p.confidence:6.2f}%  {bar}")


def _make_metadata(api_key: str) -> tuple[tuple[str, str], ...]:
    """Build gRPC call metadata with the API key."""
    return (("x-api-key", api_key),)


# ── Async request generator for streaming ────────────────────────────────────

async def _image_request_generator(
    image_paths: list[Path],
    *,
    top_k: int = 5,
) -> AsyncIterator[inference_pb2.PredictRequest]:
    """Yield one PredictRequest per image file."""
    for path in image_paths:
        image_bytes = path.read_bytes()
        yield inference_pb2.PredictRequest(
            image_data=image_bytes,
            request_id=str(uuid.uuid4()),
            top_k=top_k,
        )
        # Yield control to the event loop between requests so the stream
        # doesn't starve other coroutines.
        await asyncio.sleep(0)


# ── RPC modes ────────────────────────────────────────────────────────────────

async def cmd_predict(
    stub: inference_pb2_grpc.InferenceServiceStub,
    image_path: str,
    *,
    api_key: str,
    top_k: int,
) -> None:
    """
    Unary Predict RPC.

    Sends a single image and prints the top-K predictions with timing.
    """
    path = Path(image_path)
    if not path.is_file():
        print(f"ERROR: '{image_path}' is not a file.", file=sys.stderr)
        sys.exit(1)

    image_bytes = path.read_bytes()
    request_id  = str(uuid.uuid4())

    request = inference_pb2.PredictRequest(
        image_data=image_bytes,
        request_id=request_id,
        top_k=top_k,
    )

    t0 = time.perf_counter()
    try:
        response: inference_pb2.PredictResponse = await stub.Predict(
            request,
            metadata=_make_metadata(api_key),
            timeout=30,
        )
    except grpc.aio.AioRpcError as exc:
        print(f"gRPC error [{exc.code().name}]: {exc.details()}", file=sys.stderr)
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _print_predictions(response, image_name=path.name, elapsed_ms=elapsed_ms)


async def cmd_stream(
    stub: inference_pb2_grpc.InferenceServiceStub,
    image_dir: str,
    *,
    api_key: str,
    top_k: int,
) -> None:
    """
    Bidirectional streaming PredictStream RPC.

    Sends all JPEG/PNG/BMP/WebP files in the given directory as a stream
    and prints results as they arrive. Useful for batch scoring a folder
    of test images without the per-RPC overhead of individual Predict calls.
    """
    directory = Path(image_dir)
    if not directory.is_dir():
        print(f"ERROR: '{image_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES
    )

    if not image_paths:
        print(f"No supported images found in '{image_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming {len(image_paths)} image(s) from '{image_dir}' …\n")

    n_ok = 0
    n_err = 0
    t_start = time.perf_counter()

    try:
        call = stub.PredictStream(
            _image_request_generator(image_paths, top_k=top_k),
            metadata=_make_metadata(api_key),
            timeout=120,
        )
        async for response in call:
            if response.error:
                n_err += 1
                print(f"  [ERROR] {response.request_id}: {response.error}")
            else:
                n_ok += 1
                top = response.predictions[0] if response.predictions else None
                label = f"{top.label} ({top.confidence:.2f}%)" if top else "—"
                print(
                    f"  [{n_ok + n_err:>4}] {response.worker_id}  "
                    f"{response.inference_time_ms:6.1f} ms  {label}"
                )
    except grpc.aio.AioRpcError as exc:
        print(f"\ngRPC stream error [{exc.code().name}]: {exc.details()}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t_start
    throughput = (n_ok + n_err) / elapsed if elapsed > 0 else 0.0

    print(
        f"\nStream complete: {n_ok} OK  |  {n_err} errors  |  "
        f"{elapsed:.2f} s  |  {throughput:.1f} req/s"
    )


async def cmd_health(
    stub: inference_pb2_grpc.InferenceServiceStub,
    *,
    api_key: str,  # noqa: ARG001  — HealthCheck intentionally requires no auth
) -> None:
    """
    Unary HealthCheck RPC — no API key required by the server.
    """
    try:
        response: inference_pb2.HealthCheckResponse = await stub.HealthCheck(
            inference_pb2.HealthCheckRequest(),
            timeout=5,
        )
    except grpc.aio.AioRpcError as exc:
        print(f"gRPC error [{exc.code().name}]: {exc.details()}", file=sys.stderr)
        sys.exit(1)

    status = "✓ HEALTHY" if response.healthy else "✗ UNHEALTHY"
    print(f"\nHealth check: {status}")
    print(f"  worker_id       : {response.worker_id}")
    print(f"  device          : {response.device}")
    print(f"  queue_depth     : {response.queue_depth}")
    print(f"  p95_latency_ms  : {response.p95_latency_ms:.1f} ms")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="gRPC Inference Client — Distributed DL Inference Server v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="gRPC server host (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="gRPC server port (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for authentication (default: dev-key-abc123)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to return (default: 5)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p_predict = sub.add_parser("predict", help="Single unary inference RPC")
    p_predict.add_argument("image", help="Path to image file (JPEG/PNG/BMP/WebP)")

    # stream
    p_stream = sub.add_parser(
        "stream", help="Bidirectional streaming RPC over a directory of images"
    )
    p_stream.add_argument(
        "image_dir",
        help="Directory containing JPEG/PNG/BMP/WebP files",
    )

    # health
    sub.add_parser("health", help="HealthCheck RPC (no auth required)")

    args = parser.parse_args()

    target = f"{args.host}:{args.port}"

    channel_opts = [
        ("grpc.max_receive_message_length", 50 * 1024 * 1024 + 1024),
        ("grpc.max_send_message_length",    50 * 1024 * 1024 + 1024),
        ("grpc.keepalive_time_ms",               30_000),
        ("grpc.keepalive_timeout_ms",            10_000),
        ("grpc.keepalive_permit_without_calls",       1),
    ]

    async with grpc.aio.insecure_channel(target, options=channel_opts) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        if args.command == "predict":
            await cmd_predict(
                stub,
                args.image,
                api_key=args.api_key,
                top_k=args.top_k,
            )

        elif args.command == "stream":
            await cmd_stream(
                stub,
                args.image_dir,
                api_key=args.api_key,
                top_k=args.top_k,
            )

        elif args.command == "health":
            await cmd_health(stub, api_key=args.api_key)


if __name__ == "__main__":
    asyncio.run(main())
