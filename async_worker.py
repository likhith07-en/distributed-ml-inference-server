#!/usr/bin/env python3
"""
async_worker.py — Production-grade async inference worker
Handles 10,000+ concurrent connections; single-threaded event loop for I/O;
ProcessPoolExecutor for CPU preprocessing; GPU inference serialized.
"""
import asyncio, uvloop, struct, json, logging, time, os
from concurrent.futures import ProcessPoolExecutor
from model_loader import ModelLoader
from batch_engine import DynamicBatchEngine
from security import validate_image_bytes
from logging_config import setup_logging

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

MAX_PAYLOAD   = 50 * 1024 * 1024   # 50 MB hard cap
PREPROCESS_W  = 4                   # CPU pool workers
INFER_SEM     = 1                   # Serialize GPU calls

model:    ModelLoader        = None
engine:   DynamicBatchEngine = None
sem:      asyncio.Semaphore  = None
cpu_pool: ProcessPoolExecutor = None

async def recv_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    return await reader.readexactly(n)  # raises IncompleteReadError on disconnect

async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    t0   = time.perf_counter()
    try:
        # 1. Read 8-byte length-prefixed header
        raw_hdr     = await recv_exactly(reader, 8)
        payload_len = struct.unpack(">Q", raw_hdr)[0]
        if payload_len == 0 or payload_len > MAX_PAYLOAD:
            raise ValueError(f"Illegal payload length: {payload_len}")

        # 2. Read image data (non-blocking)
        image_bytes = await recv_exactly(reader, payload_len)

        # 3. Validate image format (magic bytes)
        validate_image_bytes(image_bytes)

        # 4. CPU preprocessing in process pool (bypasses GIL)
        loop   = asyncio.get_running_loop()
        img_np = await loop.run_in_executor(cpu_pool,
                                            model.preprocess_to_numpy,
                                            image_bytes)

        # 5. Dynamic batching — submit to engine, await Future
        preds = await engine.submit(img_np)

        # 6. Send response
        resp = json.dumps(preds).encode()
        writer.write(struct.pack(">Q", len(resp)) + resp)
        await writer.drain()

        ms = (time.perf_counter() - t0) * 1000
        logging.getLogger("worker").info(
            "request complete",
            extra={"peer": str(peer), "latency_ms": round(ms,1),
                   "top": preds[0]["label"]})

    except asyncio.IncompleteReadError:
        logging.getLogger("worker").warning(f"{peer} disconnected mid-transfer")
    except ValueError as e:
        logging.getLogger("worker").error(f"{peer} validation: {e}")
    except Exception as e:
        logging.getLogger("worker").exception(f"{peer} error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()

async def main(port: int = int(os.getenv("WORKER_PORT", "10001"))) -> None:
    global model, engine, sem, cpu_pool
    setup_logging()
    model    = ModelLoader()
    model.warmup()
    sem      = asyncio.Semaphore(INFER_SEM)
    cpu_pool = ProcessPoolExecutor(max_workers=PREPROCESS_W)
    engine   = DynamicBatchEngine(model)
    await engine.start()
    server   = await asyncio.start_server(
        handle_client, "0.0.0.0", port, limit=2**20, backlog=4096)
    logging.getLogger("worker").info(f"Async worker on :{port}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
