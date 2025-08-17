"""
State transition validation for the bot's state machine.

This module provides the StateTransitionValidator class which enforces
valid state transitions and prevents invalid state changes that could
lead to bugs, race conditions, or inconsistent behavior.
"""

from .state import BotStateEnum
from src.exceptions import StateTransitionError


class StateTransitionValidator:
    """
    Validates state transitions for the bot's state machine.

    This class contains the authoritative rules for valid state transitions
    and enforces them to prevent invalid state changes.
    """

    # Map of valid state transitions: from_state -> {valid_to_states}
    _VALID_TRANSITIONS = {
        BotStateEnum.IDLE: {
            BotStateEnum.STANDBY,  # Manual connect
            BotStateEnum.LISTENING,  # Realtime connect
        },
        BotStateEnum.STANDBY: {
            BotStateEnum.RECORDING,  # PTT/wake word activated
            BotStateEnum.IDLE,  # Disconnect
            BotStateEnum.CONNECTION_ERROR,  # Connection issues
        },
        BotStateEnum.RECORDING: {
            BotStateEnum.STANDBY,  # Recording finished
            BotStateEnum.CONNECTION_ERROR,  # Connection issues
        },
        BotStateEnum.LISTENING: {
            BotStateEnum.SPEAKING,  # AI starts talking
            BotStateEnum.IDLE,  # Disconnect
            BotStateEnum.CONNECTION_ERROR,  # Connection issues
        },
        BotStateEnum.SPEAKING: {
            BotStateEnum.LISTENING,  # AI stops talking
            BotStateEnum.CONNECTION_ERROR,  # Connection issues
        },
        BotStateEnum.CONNECTION_ERROR: {
            BotStateEnum.STANDBY,  # Recovery to manual mode
            BotStateEnum.LISTENING,  # Recovery to realtime mode
            BotStateEnum.IDLE,  # Disconnect/give up
        },
    }

    @classmethod
    def validate(cls, from_state: BotStateEnum, to_state: BotStateEnum) -> None:
        """
        Validates a state transition and raises an error if it's invalid.

        Args:
            from_state: The current state to transition from
            to_state: The desired state to transition to

        Raises:
            StateTransitionError: If the transition is not allowed
        """
        # Allow self-transitions (no-op transitions are handled by caller)
        if from_state == to_state:
            return

        valid_to_states = cls._VALID_TRANSITIONS.get(from_state, set())

        if to_state not in valid_to_states:
            raise StateTransitionError(
                f"Invalid state transition from {from_state.value} to {to_state.value}. "
                f"Valid transitions from {from_state.value}: {[s.value for s in valid_to_states]}"
            )

    @classmethod
    def get_valid_transitions(cls, from_state: BotStateEnum) -> set[BotStateEnum]:
        """
        Get the set of valid states that can be transitioned to from the given state.

        Args:
            from_state: The state to get valid transitions from

        Returns:
            Set of valid target states
        """
        return cls._VALID_TRANSITIONS.get(from_state, set()).copy()
