# shutdown.py — Graceful shutdown with connection draining
# Kubernetes terminationGracePeriodSeconds should be DRAIN_TIMEOUT_SEC + 5.

import asyncio, signal, logging

log = logging.getLogger("shutdown")

class GracefulShutdown:
    DRAIN_TIMEOUT_SEC = 30

    def __init__(self, server: asyncio.Server):
        self.server    = server
        self._shutdown = asyncio.Event()

    def install(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

    def _handle_signal(self) -> None:
        log.info("Shutdown signal received — draining connections",
                 extra={"timeout_sec": self.DRAIN_TIMEOUT_SEC})
        self._shutdown.set()
        self.server.close()

    async def wait(self) -> None:
        await self._shutdown.wait()
        log.info("Waiting for in-flight requests to complete ...")
        try:
            await asyncio.wait_for(
                self.server.wait_closed(),
                timeout=self.DRAIN_TIMEOUT_SEC)
            log.info("Graceful shutdown complete")
        except asyncio.TimeoutError:
            log.warning("Drain timeout — forcing exit")
