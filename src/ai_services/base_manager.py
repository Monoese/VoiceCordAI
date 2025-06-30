"""
Defines the BaseRealtimeManager class, an abstract base for real-time AI service managers.

This class uses the Template Method design pattern to encapsulate common logic for
connection management, configuration, and state checking, while deferring
provider-specific implementation details to subclasses.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, Protocol

from src.ai_services.interface import IRealtimeAIServiceManager
from src.audio.playback import AudioPlaybackManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Using a Protocol to define the expected interface for connection handlers.
# This avoids a circular dependency if we were to import the concrete connection classes.
class ConnectionHandler(Protocol):
    """Defines the interface for connection handler classes."""

    async def connect(
        self, event_callback: Callable[[Any], Awaitable[None]]
    ) -> None: ...

    async def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    async def wait_until_connected(self) -> None: ...


class BaseRealtimeManager(IRealtimeAIServiceManager):
    """
    Abstract base class for real-time AI service managers.

    This class provides a template for managing AI service interactions,
    handling common tasks like connection, disconnection, and configuration.
    Subclasses must implement abstract properties and methods to provide
    provider-specific logic.
    """

    def __init__(
        self,
        audio_playback_manager: AudioPlaybackManager,
        service_config: Dict[str, Any],
    ) -> None:
        """
        Initializes the BaseRealtimeManager.

        Args:
            audio_playback_manager: An instance of AudioPlaybackManager.
            service_config: Configuration specific to the service instance.

        Raises:
            ValueError: If API key or model name is missing in the configuration.
        """
        super().__init__(audio_playback_manager, service_config)
        self._api_key: str = self._service_config.get("api_key")
        if not self._api_key:
            raise ValueError("API key is missing in the service configuration.")

        self._model_name: str = self._service_config.get("model_name")
        if not self._model_name:
            raise ValueError("Model name is missing in the service configuration.")

    @property
    @abstractmethod
    def _connection_handler(self) -> ConnectionHandler:
        """
        Abstract property for the provider-specific connection handler.
        Subclasses must return an object that conforms to the ConnectionHandler protocol.
        """
        pass

    @property
    @abstractmethod
    def _event_callback(self) -> Callable[[Any], Awaitable[None]]:
        """
        Abstract property for the provider-specific event callback.
        Subclasses must return the callable that dispatches events from the connection.
        """
        pass

    @abstractmethod
    async def _post_connect_hook(self) -> bool:
        """
        Abstract hook for post-connection logic.
        This is called after a connection is established.
        Subclasses can implement this to perform provider-specific setup
        (e.g., sending initial session data).

        Returns:
            True if successful, False on failure. A False return will trigger a disconnect.
        """
        pass

    async def connect(self) -> bool:
        """
        Establishes a connection using the provider-specific connection handler.
        This method implements the common connection logic.

        Returns:
            True if the connection was successful, False otherwise.
        """
        manager_name = self.__class__.__name__
        logger.info(f"{manager_name}: Attempting to connect...")
        if self.is_connected():
            logger.info(f"{manager_name}: Already connected.")
            return True

        timeout = self._service_config.get("connection_timeout", 30.0)
        try:
            await self._connection_handler.connect(self._event_callback)

            await asyncio.wait_for(
                self._connection_handler.wait_until_connected(), timeout=timeout
            )
            logger.info(
                f"{manager_name}: Connection successfully established with handler."
            )

            if not await self._post_connect_hook():
                logger.error(
                    f"{manager_name}: Post-connect hook failed. Disconnecting."
                )
                await self.disconnect()
                return False

            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"{manager_name}: Failed to establish connection within {timeout}s timeout."
            )
            await self.disconnect()
            return False
        except Exception as e:
            logger.error(
                f"{manager_name}: An unexpected error occurred during connection: {e}",
                exc_info=True,
            )
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """
        Closes the connection using the provider-specific connection handler.

        Returns:
            None.
        """
        manager_name = self.__class__.__name__
        logger.info(f"{manager_name}: Attempting to disconnect...")
        await self._connection_handler.disconnect()
        logger.info(f"{manager_name}: Disconnected.")

    def is_connected(self) -> bool:
        """
        Checks connection status by delegating to the provider-specific connection handler.

        Returns:
            True if connected, False otherwise.
        """
        # Ensure connection handler is available before checking status.
        # It might not be initialized if the subclass __init__ fails.
        try:
            return self._connection_handler.is_connected()
        except AttributeError:
            return False
