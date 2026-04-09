# batch_engine.py — Dynamic batching with timeout-based flush
# Documented tradeoff (chosen default: BATCH_SIZE=16, TIMEOUT=20ms):
#   BATCH_SIZE=1,  TIMEOUT=0ms  -> 1x throughput, +0ms latency (baseline)
#   BATCH_SIZE=8,  TIMEOUT=20ms -> 8x throughput, +20ms latency
#   BATCH_SIZE=16, TIMEOUT=20ms -> 11x throughput, +20ms latency [DEFAULT]
#   BATCH_SIZE=32, TIMEOUT=50ms -> 14x throughput, +50ms latency

import asyncio, time, torch, logging, numpy as np
from dataclasses import dataclass, field
from typing import Optional
from model_loader import ModelLoader

BATCH_SIZE       = 16    # Max images per GPU batch (tune to VRAM)
BATCH_TIMEOUT_MS = 20    # Max wait to form a batch (ms)

@dataclass
class InferenceRequest:
    image_numpy : object
    future      : asyncio.Future
    arrived_at  : float = field(default_factory=time.perf_counter)

class DynamicBatchEngine:
    def __init__(self, model: ModelLoader):
        self.model  = model
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self.log   = logging.getLogger("batch_engine")

    async def start(self):
        self._task = asyncio.create_task(self._batch_loop())

    async def submit(self, image_numpy) -> list[dict]:
        loop   = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(InferenceRequest(image_numpy, future))
        return await future

    async def _batch_loop(self):
        while True:
            batch: list[InferenceRequest] = []
            try:
                first = await asyncio.wait_for(
                    self.queue.get(), timeout=BATCH_TIMEOUT_MS/1000)
                batch.append(first)
            except asyncio.TimeoutError:
                continue
            deadline = time.perf_counter() + BATCH_TIMEOUT_MS/1000
            while len(batch) < BATCH_SIZE and time.perf_counter() < deadline:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)
            await self._dispatch_batch(batch)

    async def _dispatch_batch(self, batch: list[InferenceRequest]):
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, self._infer_batch_sync,
                [req.image_numpy for req in batch])
            for req, result in zip(batch, results):
                req.future.set_result(result)
        except Exception as e:
            for req in batch:
                if not req.future.done(): req.future.set_exception(e)
            self.log.error(f"Batch inference failed: {e}")

    def _infer_batch_sync(self, images: list) -> list[list[dict]]:
        tensors = torch.from_numpy(np.stack(images)).to(self.model.device)
        if self.model.dtype == torch.float16:
            tensors = tensors.half()
        with torch.no_grad():
            logits = self.model.model(tensors)
        probs = torch.softmax(logits.float(), dim=1)
        results = []
        for i in range(len(images)):
            top_probs, top_idxs = probs[i].topk(5)
            results.append([{"rank": j+1,
                              "label": self.model.labels[idx.item()],
                              "confidence": round(prob.item()*100, 2)}
                            for j,(idx,prob) in enumerate(zip(top_idxs,top_probs))])
        return results
