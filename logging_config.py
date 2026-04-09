# logging_config.py — Structured JSON logging
# Replaces every print() call in the codebase.
# Output: {"timestamp":...,"level":...,"logger":...,"message":...,"latency_ms":...}

import logging, json, time, os, sys
from typing import Any

class JSONFormatter(logging.Formatter):
    RESERVED = {"message","timestamp","level","logger","worker_id"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(record.created)),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "worker_id": os.getenv("WORKER_ID", "unknown"),
            "pid":       record.process,
        }
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord.__dict__ and k not in self.RESERVED:
                payload[k] = v
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers = [handler]
