import asyncio
import json
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from src.config.config import Config
from src.websocket.events.events import BaseEvent
from src.utils.logger import get_logger

log = get_logger(__name__)


class WebSocketManager:
    def __init__(self) -> None:
        self._url: str = Config.WS_SERVER_URL
        self._headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1", }

        self._incoming: asyncio.Queue[BaseEvent] = asyncio.Queue()
        self._outgoing: asyncio.Queue[BaseEvent] = asyncio.Queue()

        self._ws: Optional[websockets.client.ClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._main_task: Optional[asyncio.Task] = None

        self._running = asyncio.Event()
        self._running.clear()

    async def start(self) -> None:
        """Start the background-reconnecting task (idempotent)."""
        if self._main_task and not self._main_task.done():
            return

        self._running.set()
        self._main_task = asyncio.create_task(self._connect_forever())

    async def stop(self) -> None:
        """Signal all loops to finish and wait for a clean shutdown."""
        self._running.clear()

        if self._ws:
            await self._ws.close()

        for task in (self._receive_task, self._send_task, self._main_task):
            if task and not task.done():
                task.cancel()

        await asyncio.gather(*(t for t in (self._receive_task, self._send_task, self._main_task) if t),
            return_exceptions=True, )

        self._ws = None
        self._receive_task = None
        self._send_task = None
        self._main_task = None

    async def close(self) -> None:
        await self.stop()

    async def send_event(self, event: BaseEvent) -> None:
        """Queue an event to be sent to the server."""
        await self._outgoing.put(event)

    async def get_next_event(self) -> BaseEvent:
        """Retrieve the next inbound event (await)."""
        return await self._incoming.get()

    def task_done(self) -> None:
        """Mark the last processed incoming event as done."""
        self._incoming.task_done()

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    @property
    def connection(self):
        return self._ws

    async def _connect_forever(self) -> None:
        """Reconnect with exponential back‑off until ``stop`` is called."""
        backoff = 1
        while self._running.is_set():
            try:
                await self._connect_once()
                backoff = 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("WebSocket error: %s – reconnecting in %ss", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect_once(self) -> None:
        """Open a single WebSocket session; returns when it closes."""
        async with websockets.connect(self._url, additional_headers=self._headers) as ws:
            log.info("Connected to WebSocket server → %s", self._url)
            self._ws = ws

            self._receive_task = asyncio.create_task(self._receive_loop())
            self._send_task = asyncio.create_task(self._send_loop())

            done, pending = await asyncio.wait((self._receive_task, self._send_task),
                return_when=asyncio.FIRST_EXCEPTION, )

            for task in pending:
                task.cancel()

            for task in done:
                task.result()

        self._ws = None

    async def _receive_loop(self) -> None:
        """Background task: convert raw JSON → ``BaseEvent`` → queue."""
        assert self._ws is not None
        async for message in self._ws:
            try:
                event_dict = json.loads(message)
                event = BaseEvent.from_json(event_dict)
                if event:
                    await self._incoming.put(event)
                else:
                    log.debug("Dropping unknown event type: %s", event_dict.get("type"))
            except Exception as exc:
                log.exception("Failed to parse message: %s", exc)

    async def _send_loop(self) -> None:
        """Background task: pop events from queue and transmit."""
        assert self._ws is not None
        while True:
            event: BaseEvent = await self._outgoing.get()
            try:
                await self._ws.send(event.to_json())
                log.debug("Sent event: %s", event.type)
            except ConnectionClosed as cexc:
                log.warning("Connection closed while sending: %s", cexc)
                raise
            finally:
                self._outgoing.task_done()
