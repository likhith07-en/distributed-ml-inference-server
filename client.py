#!/usr/bin/env python3
"""
client.py — REST Inference Client
Sends image to POST /predict; prints top-5 predictions.
Usage: python client.py <image.jpg> [--host localhost] [--port 8000]
"""
import requests, sys, time, argparse

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 8000
API_KEY      = "dev-key-abc123"

def send_image(path: str, host: str = GATEWAY_HOST,
               port: int = GATEWAY_PORT) -> list[dict]:
    url = f"http://{host}:{port}/predict"
    with open(path, "rb") as f:
        resp = requests.post(url, files={"file": f},
                             headers={"X-Api-Key": API_KEY}, timeout=30)
    resp.raise_for_status()
    return resp.json()["predictions"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--host", default=GATEWAY_HOST)
    ap.add_argument("--port", type=int, default=GATEWAY_PORT)
    args = ap.parse_args()
    t0   = time.perf_counter()
    preds = send_image(args.image, args.host, args.port)
    ms    = (time.perf_counter() - t0) * 1000
    print(f"\nTop Predictions for '{args.image}' [{ms:.1f}ms]:")
    for p in preds:
        bar = '█' * int(p['confidence'] / 2)
        print(f"  {p['rank']}. {p['label']:<35} {p['confidence']:6.2f}% {bar}")

if __name__ == "__main__":
    main()
