#!/usr/bin/env python3
"""
benchmark.py — Production-grade load testing (MLPerf methodology).
pip install aiohttp numpy scipy
Run: python benchmark.py --url http://localhost:8000/predict --images ./test_images/
"""
from __future__ import annotations
import asyncio, aiohttp, time, argparse
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class BenchmarkResult:
    concurrency: int
    n_requests:  int
    n_errors:    int
    latencies:   list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.n_errors / self.n_requests * 100

    @property
    def throughput(self) -> float:
        return (len(self.latencies) / (max(self.latencies)/1000)
                if self.latencies else 0)

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.latencies, p)) if self.latencies else 0

    def confidence_interval(self, p: float=0.95) -> tuple[float,float]:
        if len(self.latencies) < 2: return (0, 0)
        ci = scipy_stats.t.interval(p, len(self.latencies)-1,
                                    loc=np.mean(self.latencies),
                                    scale=scipy_stats.sem(self.latencies))
        return (round(ci[0],1), round(ci[1],1))

async def run_one(session, url, img_path, result, lock, api_key) -> None:
    t0 = time.perf_counter()
    try:
        async with session.post(url, data=img_path.read_bytes(),
                                headers={"X-Api-Key": api_key},
                                timeout=aiohttp.ClientTimeout(total=5)) as resp:
            await resp.json()
        ms = (time.perf_counter()-t0)*1000
        async with lock: result.latencies.append(ms)
    except Exception:
        async with lock: result.n_errors += 1

async def benchmark(url, image_dir, concurrency, n, warmup=20,
                    api_key="dev-key-abc123"):
    images = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    assert images, f"No images in {image_dir}"
    result = BenchmarkResult(concurrency=concurrency, n_requests=n, n_errors=0)
    lock   = asyncio.Lock()
    conn   = aiohttp.TCPConnector(limit=concurrency, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=conn) as session:
        print(f"  Warming up ({warmup} requests) ...")
        dummy = BenchmarkResult(1, warmup, 0)
        await asyncio.gather(*[run_one(session, url, images[i%len(images)],
                                       dummy, asyncio.Lock(), api_key)
                                for i in range(warmup)])
        await asyncio.gather(*[run_one(session, url, images[i%len(images)],
                                       result, lock, api_key)
                                for i in range(n)])
    return result

def print_report(r: BenchmarkResult) -> None:
    ci = r.confidence_interval()
    print(f"  Concurrency : {r.concurrency}")
    print(f"  Requests    : {r.n_requests} | Errors: {r.n_errors} ({r.error_rate:.1f}%)")
    print(f"  Throughput  : {r.throughput:.1f} req/s")
    print(f"  P50         : {r.percentile(50):.1f} ms")
    print(f"  P95         : {r.percentile(95):.1f} ms")
    print(f"  P99         : {r.percentile(99):.1f} ms")
    print(f"  P99.9       : {r.percentile(99.9):.1f} ms")
    print(f"  95% CI(mean): {ci[0]:.1f} -- {ci[1]:.1f} ms")
    print(f"  Max         : {max(r.latencies, default=0):.1f} ms")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",    default="http://localhost:8000/predict")
    ap.add_argument("--images", default="./test_images/")
    ap.add_argument("--n",      type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    for c in [1, 4, 8, 16, 32, 64]:
        print(f"\n── Concurrency={c} ──────────────────")
        r = asyncio.run(benchmark(args.url, args.images, c, args.n, args.warmup))
        print_report(r)
