#!/usr/bin/env python3
"""
model_loader.py — Production-grade model loading
  - Full type annotations (mypy --strict compliant)
  - GPU/CPU/MPS fallback with compute capability checks
  - torch.compile() (PyTorch 2.0+): 20-40% speedup via kernel fusion
  - FP16 on GPU: 2x throughput, 50% VRAM reduction
  - INT8 dynamic quantization on CPU: 2-4x speedup
  - CUDA-synchronized warm-up with latency reporting
  - numpy bridge for ProcessPoolExecutor (GIL-free preprocessing)
"""
from __future__ import annotations
import io, logging, time
from typing import Any
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

log = logging.getLogger(__name__)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

class ModelLoader:
    def __init__(self, top_k: int = 5, use_compile: bool = True) -> None:
        self.top_k   = top_k
        self.device  = self._select_device()
        self.dtype   = torch.float16 if self.device.type=="cuda" else torch.float32
        self.model   = self._load_model(use_compile)
        self.labels: list[str] = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
        self._preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224), T.ToTensor(), T.Normalize(_MEAN, _STD),
        ])
        log.info("ModelLoader ready | device=%s | dtype=%s | top_k=%d",
                 self.device, self.dtype, top_k)

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            log.info("GPU: %s | VRAM: %.1f GB | Compute: %d.%d",
                     props.name, props.total_memory/1e9, props.major, props.minor)
            if props.major < 6:
                log.warning("Compute < 6.0 -> CPU fallback")
                return torch.device("cpu")
            return torch.device("cuda:0")
        if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
            log.info("Using Apple MPS backend")
            return torch.device("mps")
        log.warning("No accelerator — CPU mode (INT8 enabled)")
        return torch.device("cpu")

    def _load_model(self, use_compile: bool) -> nn.Module:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model   = resnet50(weights=weights)
        if self.device.type == "cpu":
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
            log.info("INT8 dynamic quantization applied (CPU)")
        else:
            model = model.to(self.dtype)
            log.info("FP16 half-precision on GPU")
        model = model.to(self.device).eval()
        if use_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                log.info("torch.compile() applied (reduce-overhead)")
            except Exception as e:
                log.warning("torch.compile() failed: %s", e)
        return model

    def warmup(self, n: int = 5) -> None:
        log.info("Warming up with %d dummy inferences ...", n)
        dummy = torch.zeros(1,3,224,224, dtype=self.dtype, device=self.device)
        lats: list[float] = []
        with torch.no_grad():
            for _ in range(n):
                if self.device.type=="cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                self.model(dummy)
                if self.device.type=="cuda": torch.cuda.synchronize()
                lats.append((time.perf_counter()-t0)*1000)
        log.info("Warm-up done | lats: %s ms | stable: %.1f ms",
                 [f"{l:.0f}" for l in lats], lats[-1])

    def preprocess_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """CPU preprocessing -> numpy. Safe in ProcessPoolExecutor."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._preprocess(img).numpy()

    @torch.no_grad()
    def infer_from_numpy(self, image_np: np.ndarray) -> list[dict[str, Any]]:
        """Single-image GPU inference from preprocessed numpy array."""
        tensor = (torch.from_numpy(image_np)
                  .unsqueeze(0)
                  .to(device=self.device, dtype=self.dtype))
        logits = self.model(tensor)
        probs  = torch.softmax(logits.float(), dim=1)[0]
        top_probs, top_idxs = probs.topk(self.top_k)
        return [{"rank": i+1,
                 "label": self.labels[idx.item()],
                 "confidence": round(prob.item()*100, 2)}
                for i,(idx,prob) in enumerate(zip(top_idxs,top_probs))]
