import asyncio
import json
import websockets
from config import Config
from events import BaseEvent
from logger import get_logger

logger = get_logger(__name__)

class WebSocketManager:
    def __init__(self):
        self.connection = None
        self.incoming_events = asyncio.Queue()
        self.outgoing_events = asyncio.Queue()

    async def connect(self):
        """Establish WebSocket connection with proper headers"""
        headers = {
            "Authorization": "Bearer " + Config.OPENAI_API_KEY,
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            async with websockets.connect(Config.WS_SERVER_URL, additional_headers=headers) as websocket:
                self.connection = websocket
                logger.info("Connected to WebSocket server.")

                receive_task = asyncio.create_task(self._receive_events())
                send_task = asyncio.create_task(self._send_events())

                await asyncio.wait(
                    [receive_task, send_task], 
                    timeout=Config.CONNECTION_TIMEOUT
                )

                if not receive_task.done() or not send_task.done():
                    logger.warning("Connection timed out after 15 minutes. Reconnecting...")
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed. Reconnecting...")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}. Reconnecting...")
        finally:
            logger.warning("WebSocket connection failed to be recovered.")
            self.connection = None
            await asyncio.sleep(1)
            asyncio.create_task(self.connect())

    async def _receive_events(self):
        """Handles receiving events from the WebSocket server"""
        while True:
            message = await self.connection.recv()
            data = json.loads(message)

            event = BaseEvent.from_json(data)

            if event is not None:
                logger.debug(f"Received event: {event.type}")
                await self.incoming_events.put(event)
            else:
                logger.warning(f"Handler for {data['type']} not available")

    async def _send_events(self):
        """Handles sending events to the WebSocket server"""
        while True:
            event = await self.outgoing_events.get()
            try:
                await self.connection.send(event.to_json())
                logger.debug(f"Sent event: {event.type}")
            except Exception as e:
                logger.error(f"Error sending event {event.type}: {e}")
            finally:
                self.outgoing_events.task_done()

    async def send_event(self, event):
        """Add an event to the outgoing queue"""
        await self.outgoing_events.put(event)

    async def get_next_event(self):
        """Get the next event from the incoming queue"""
        return await self.incoming_events.get()

    def task_done(self):
        """Mark the current task as done"""
        self.incoming_events.task_done()

    async def close(self):
        """Close the WebSocket connection"""
        if self.connection:
            await self.connection.close()
            self.connection = None
