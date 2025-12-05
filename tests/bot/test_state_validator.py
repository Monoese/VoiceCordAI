"""
Unit tests for the StateTransitionValidator.

This module tests the state transition validation logic to ensure
that only valid state transitions are allowed and invalid ones
are properly rejected with appropriate error messages.
"""

import pytest
from src.bot.state import BotStateEnum
from src.bot.state_validator import StateTransitionValidator
from src.exceptions import StateTransitionError


class TestStateTransitionValidator:
    """Test cases for the StateTransitionValidator class."""

    def test_valid_transitions_from_idle(self):
        """Test all valid transitions from IDLE state."""
        # IDLE -> STANDBY (connect)
        StateTransitionValidator.validate(BotStateEnum.IDLE, BotStateEnum.STANDBY)

    def test_valid_transitions_from_standby(self):
        """Test all valid transitions from STANDBY state."""
        # STANDBY -> RECORDING (PTT/wake word)
        StateTransitionValidator.validate(BotStateEnum.STANDBY, BotStateEnum.RECORDING)

        # STANDBY -> IDLE (disconnect)
        StateTransitionValidator.validate(BotStateEnum.STANDBY, BotStateEnum.IDLE)

        # STANDBY -> CONNECTION_ERROR (connection issues)
        StateTransitionValidator.validate(
            BotStateEnum.STANDBY, BotStateEnum.CONNECTION_ERROR
        )

    def test_valid_transitions_from_recording(self):
        """Test all valid transitions from RECORDING state."""
        # RECORDING -> STANDBY (recording finished)
        StateTransitionValidator.validate(BotStateEnum.RECORDING, BotStateEnum.STANDBY)

        # RECORDING -> CONNECTION_ERROR (connection issues)
        StateTransitionValidator.validate(
            BotStateEnum.RECORDING, BotStateEnum.CONNECTION_ERROR
        )

    def test_valid_transitions_from_connection_error(self):
        """Test all valid transitions from CONNECTION_ERROR state."""
        # CONNECTION_ERROR -> STANDBY (recovery)
        StateTransitionValidator.validate(
            BotStateEnum.CONNECTION_ERROR, BotStateEnum.STANDBY
        )

        # CONNECTION_ERROR -> IDLE (disconnect/give up)
        StateTransitionValidator.validate(
            BotStateEnum.CONNECTION_ERROR, BotStateEnum.IDLE
        )

    def test_self_transitions_are_allowed(self):
        """Test that self-transitions (same state) are allowed."""
        for state in BotStateEnum:
            # Self-transitions should not raise an exception
            StateTransitionValidator.validate(state, state)

    def test_invalid_transitions_from_idle(self):
        """Test invalid transitions from IDLE state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(BotStateEnum.IDLE, BotStateEnum.RECORDING)
        assert "Invalid state transition from idle to recording" in str(exc_info.value)

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.IDLE, BotStateEnum.CONNECTION_ERROR
            )

    def test_invalid_transitions_from_standby(self):
        """Test invalid transitions from STANDBY state."""
        # No other invalid transitions besides what's already invalid
        pass

    def test_invalid_transitions_from_recording(self):
        """Test invalid transitions from RECORDING state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(BotStateEnum.RECORDING, BotStateEnum.IDLE)
        assert "Invalid state transition from recording to idle" in str(exc_info.value)

    def test_invalid_transitions_from_connection_error(self):
        """Test invalid transitions from CONNECTION_ERROR state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(
                BotStateEnum.CONNECTION_ERROR, BotStateEnum.RECORDING
            )
        assert "Invalid state transition from connection_error to recording" in str(
            exc_info.value
        )

    def test_error_message_includes_valid_transitions(self):
        """Test that error messages include the list of valid transitions."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(BotStateEnum.RECORDING, BotStateEnum.IDLE)

        error_message = str(exc_info.value)
        assert "Invalid state transition from recording to idle" in error_message
        assert "Valid transitions from recording:" in error_message
        assert "standby" in error_message
        assert "connection_error" in error_message

    def test_get_valid_transitions(self):
        """Test the get_valid_transitions helper method."""
        # Test IDLE state
        valid_from_idle = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.IDLE
        )
        expected_from_idle = {BotStateEnum.STANDBY}
        assert valid_from_idle == expected_from_idle

        # Test STANDBY state
        valid_from_standby = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.STANDBY
        )
        expected_from_standby = {
            BotStateEnum.RECORDING,
            BotStateEnum.IDLE,
            BotStateEnum.CONNECTION_ERROR,
        }
        assert valid_from_standby == expected_from_standby

        # Test RECORDING state
        valid_from_recording = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.RECORDING
        )
        expected_from_recording = {BotStateEnum.STANDBY, BotStateEnum.CONNECTION_ERROR}
        assert valid_from_recording == expected_from_recording

        # Test CONNECTION_ERROR state
        valid_from_error = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.CONNECTION_ERROR
        )
        expected_from_error = {
            BotStateEnum.STANDBY,
            BotStateEnum.IDLE,
        }
        assert valid_from_error == expected_from_error

    def test_get_valid_transitions_returns_copy(self):
        """Test that get_valid_transitions returns a copy, not the original set."""
        valid_transitions = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.IDLE
        )
        original_size = len(valid_transitions)

        # Modify the returned set
        valid_transitions.add(BotStateEnum.RECORDING)

        # Get a fresh copy and verify it wasn't affected
        fresh_transitions = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.IDLE
        )
        assert len(fresh_transitions) == original_size
        assert BotStateEnum.RECORDING not in fresh_transitions

    def test_all_states_have_transition_rules(self):
        """Test that all states in BotStateEnum have transition rules defined."""
        for state in BotStateEnum:
            # This should not raise a KeyError
            valid_transitions = StateTransitionValidator.get_valid_transitions(state)
            # Each state should have at least one valid transition (even if it's just to itself)
            # Note: We don't enforce this since CONNECTION_ERROR might only transition to other states
            assert isinstance(valid_transitions, set)
