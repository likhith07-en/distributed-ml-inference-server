# api_server.py — FastAPI REST interface over DynamicBatchEngine
# Run: uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1

from __future__ import annotations
import time, logging, uuid, hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from prometheus_fastapi_instrumentator import Instrumentator
from security import validate_image_bytes, VALID_KEY_HASHES, rate_limiter
from batch_engine import DynamicBatchEngine
from model_loader import ModelLoader

log = logging.getLogger(__name__)
app = FastAPI(title="Distributed Inference Server",
              description="ResNet-50 ImageNet classification with dynamic batching",
              version="3.0.0")
Instrumentator().instrument(app).expose(app)   # /metrics endpoint

engine: Optional[DynamicBatchEngine] = None

@app.on_event("startup")
async def startup() -> None:
    global engine
    model = ModelLoader()
    model.warmup()
    engine = DynamicBatchEngine(model)
    await engine.start()
    log.info("API server ready")

# ── Auth dependency ──────────────────────────────────────────────────────
async def verify_api_key(x_api_key: Annotated[str, Header()]) -> str:
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    if key_hash not in VALID_KEY_HASHES:
        raise HTTPException(status_code=401, detail="Invalid API key")
    client_id = VALID_KEY_HASHES[key_hash]
    if not rate_limiter.allow(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (10 req/s)")
    return client_id

# ── Response models ───────────────────────────────────────────────────────
class Prediction(BaseModel):
    rank:       int   = Field(..., ge=1, le=10)
    label:      str
    confidence: float = Field(..., ge=0.0, le=100.0)

class PredictResponse(BaseModel):
    request_id:       str
    predictions:      list[Prediction]
    inference_time_ms: float
    model_version:    str = "resnet50-imagenet1k-v2"

# ── Endpoints ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file:      UploadFile = File(..., description="Image (JPEG/PNG/BMP/WebP)"),
    client_id: str        = Depends(verify_api_key),
) -> PredictResponse:
    request_id = str(uuid.uuid4())
    t0         = time.perf_counter()
    image_bytes = await file.read()
    try:
        validate_image_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    image_np = engine.model.preprocess_to_numpy(image_bytes)
    preds    = await engine.submit(image_np)
    ms       = (time.perf_counter() - t0) * 1000
    log.info("predict", extra={"request_id": request_id, "client": client_id,
                                "latency_ms": round(ms,1), "top": preds[0]["label"],
                                "image_size": len(image_bytes)})
    return PredictResponse(request_id=request_id,
                           predictions=preds,
                           inference_time_ms=round(ms,1))

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "resnet50",
            "device": str(engine.model.device)}
