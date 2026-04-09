# security.py — API key auth + rate limiting + input validation
# All production inference APIs require these.
# Their absence is a P0 security gap.

import hashlib, time, logging
from collections import defaultdict

# SHA-256 hashed keys. NEVER store plaintext keys.
VALID_KEY_HASHES = {
    hashlib.sha256(b"dev-key-abc123").hexdigest():  "dev-client",
    hashlib.sha256(b"prod-key-xyz789").hexdigest(): "prod-client",
}

class TokenBucketRateLimiter:
    """
    Token bucket: allows burst up to capacity, then enforces rate req/s.
    Chosen over leaky bucket: allows short batch bursts while bounding sustained load.
    """
    def __init__(self, rate: float = 10.0, capacity: int = 20):
        self.rate     = rate
        self.capacity = capacity
        self.tokens   = defaultdict(lambda: capacity)
        self.last_ts  = defaultdict(time.monotonic)

    def allow(self, client_id: str) -> bool:
        now     = time.monotonic()
        elapsed = now - self.last_ts[client_id]
        self.tokens[client_id] = min(
            self.capacity,
            self.tokens[client_id] + elapsed * self.rate)
        self.last_ts[client_id] = now
        if self.tokens[client_id] >= 1:
            self.tokens[client_id] -= 1
            return True
        return False

rate_limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)

MAX_PAYLOAD_BYTES = 20 * 1024 * 1024   # 20 MB
MIN_PAYLOAD_BYTES = 100                 # minimum valid JPEG header
VALID_MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG":      "PNG",
    b"BM":           "BMP",
    b"RIFF":         "WEBP",
}

def validate_image_bytes(data: bytes) -> str:
    """Returns format string or raises ValueError with reason."""
    if len(data) < MIN_PAYLOAD_BYTES:
        raise ValueError(f"Payload too small: {len(data)} bytes")
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Payload too large: {len(data)} bytes (max {MAX_PAYLOAD_BYTES})")
    for magic, fmt in VALID_MAGIC.items():
        if data[:len(magic)] == magic:
            return fmt
    raise ValueError("Unknown format. Expected JPEG, PNG, BMP, or WEBP.")
