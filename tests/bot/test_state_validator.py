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
        # IDLE -> STANDBY (manual connect)
        StateTransitionValidator.validate(BotStateEnum.IDLE, BotStateEnum.STANDBY)

        # IDLE -> LISTENING (realtime connect)
        StateTransitionValidator.validate(BotStateEnum.IDLE, BotStateEnum.LISTENING)

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

    def test_valid_transitions_from_listening(self):
        """Test all valid transitions from LISTENING state."""
        # LISTENING -> SPEAKING (AI starts talking)
        StateTransitionValidator.validate(BotStateEnum.LISTENING, BotStateEnum.SPEAKING)

        # LISTENING -> IDLE (disconnect)
        StateTransitionValidator.validate(BotStateEnum.LISTENING, BotStateEnum.IDLE)

        # LISTENING -> CONNECTION_ERROR (connection issues)
        StateTransitionValidator.validate(
            BotStateEnum.LISTENING, BotStateEnum.CONNECTION_ERROR
        )

    def test_valid_transitions_from_speaking(self):
        """Test all valid transitions from SPEAKING state."""
        # SPEAKING -> LISTENING (AI stops talking)
        StateTransitionValidator.validate(BotStateEnum.SPEAKING, BotStateEnum.LISTENING)

        # SPEAKING -> CONNECTION_ERROR (connection issues)
        StateTransitionValidator.validate(
            BotStateEnum.SPEAKING, BotStateEnum.CONNECTION_ERROR
        )

    def test_valid_transitions_from_connection_error(self):
        """Test all valid transitions from CONNECTION_ERROR state."""
        # CONNECTION_ERROR -> STANDBY (recovery to manual mode)
        StateTransitionValidator.validate(
            BotStateEnum.CONNECTION_ERROR, BotStateEnum.STANDBY
        )

        # CONNECTION_ERROR -> LISTENING (recovery to realtime mode)
        StateTransitionValidator.validate(
            BotStateEnum.CONNECTION_ERROR, BotStateEnum.LISTENING
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
            StateTransitionValidator.validate(BotStateEnum.IDLE, BotStateEnum.SPEAKING)

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.IDLE, BotStateEnum.CONNECTION_ERROR
            )

    def test_invalid_transitions_from_standby(self):
        """Test invalid transitions from STANDBY state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(
                BotStateEnum.STANDBY, BotStateEnum.LISTENING
            )
        assert "Invalid state transition from standby to listening" in str(
            exc_info.value
        )

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.STANDBY, BotStateEnum.SPEAKING
            )

    def test_invalid_transitions_from_recording(self):
        """Test invalid transitions from RECORDING state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(BotStateEnum.RECORDING, BotStateEnum.IDLE)
        assert "Invalid state transition from recording to idle" in str(exc_info.value)

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.RECORDING, BotStateEnum.LISTENING
            )

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.RECORDING, BotStateEnum.SPEAKING
            )

    def test_invalid_transitions_from_listening(self):
        """Test invalid transitions from LISTENING state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(
                BotStateEnum.LISTENING, BotStateEnum.STANDBY
            )
        assert "Invalid state transition from listening to standby" in str(
            exc_info.value
        )

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.LISTENING, BotStateEnum.RECORDING
            )

    def test_invalid_transitions_from_speaking(self):
        """Test invalid transitions from SPEAKING state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(BotStateEnum.SPEAKING, BotStateEnum.IDLE)
        assert "Invalid state transition from speaking to idle" in str(exc_info.value)

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.SPEAKING, BotStateEnum.STANDBY
            )

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.SPEAKING, BotStateEnum.RECORDING
            )

    def test_invalid_transitions_from_connection_error(self):
        """Test invalid transitions from CONNECTION_ERROR state."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(
                BotStateEnum.CONNECTION_ERROR, BotStateEnum.RECORDING
            )
        assert "Invalid state transition from connection_error to recording" in str(
            exc_info.value
        )

        with pytest.raises(StateTransitionError):
            StateTransitionValidator.validate(
                BotStateEnum.CONNECTION_ERROR, BotStateEnum.SPEAKING
            )

    def test_error_message_includes_valid_transitions(self):
        """Test that error messages include the list of valid transitions."""
        with pytest.raises(StateTransitionError) as exc_info:
            StateTransitionValidator.validate(
                BotStateEnum.STANDBY, BotStateEnum.LISTENING
            )

        error_message = str(exc_info.value)
        assert "Invalid state transition from standby to listening" in error_message
        assert "Valid transitions from standby:" in error_message
        assert "recording" in error_message
        assert "idle" in error_message
        assert "connection_error" in error_message

    def test_get_valid_transitions(self):
        """Test the get_valid_transitions helper method."""
        # Test IDLE state
        valid_from_idle = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.IDLE
        )
        expected_from_idle = {BotStateEnum.STANDBY, BotStateEnum.LISTENING}
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

        # Test LISTENING state
        valid_from_listening = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.LISTENING
        )
        expected_from_listening = {
            BotStateEnum.SPEAKING,
            BotStateEnum.IDLE,
            BotStateEnum.CONNECTION_ERROR,
        }
        assert valid_from_listening == expected_from_listening

        # Test SPEAKING state
        valid_from_speaking = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.SPEAKING
        )
        expected_from_speaking = {BotStateEnum.LISTENING, BotStateEnum.CONNECTION_ERROR}
        assert valid_from_speaking == expected_from_speaking

        # Test CONNECTION_ERROR state
        valid_from_error = StateTransitionValidator.get_valid_transitions(
            BotStateEnum.CONNECTION_ERROR
        )
        expected_from_error = {
            BotStateEnum.STANDBY,
            BotStateEnum.LISTENING,
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
