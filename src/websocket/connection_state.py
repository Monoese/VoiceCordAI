"""
Connection state module for WebSocket connections.

This module defines the ConnectionState enum which represents the different states
a WebSocket connection can be in. This is used by the WebSocketManager to implement
a state machine for tracking and managing connection states.
"""

from enum import Enum, auto


class ConnectionState(Enum):
    """
    Enumeration of possible states for a WebSocket connection.

    These states are used to track and manage the lifecycle of
    a WebSocket connection, from initial connection attempts
    through active communication to disconnection.

    States:
    - DISCONNECTED: The connection is not established
    - CONNECTING: The connection is being established
    - CONNECTED: The connection is established and active
    - RECONNECTING: The connection was lost and is being re-established
    - DISCONNECTING: The connection is in the process of being closed
    - ERROR: The connection encountered an error
    """

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    DISCONNECTING = auto()
    ERROR = auto()
