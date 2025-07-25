"""
Custom exception hierarchy for VoiceCordAI application.

This module defines a structured exception hierarchy to replace generic exceptions
throughout the codebase, making error handling more precise and maintainable.
"""

from typing import Optional


class VoiceCordAIError(Exception):
    """Base exception for all VoiceCordAI application-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.original_error = original_error

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(VoiceCordAIError):
    """Raised when there are configuration-related errors."""

    pass


class AudioProcessingError(VoiceCordAIError):
    """Raised when audio processing operations fail."""

    pass


class AIServiceError(VoiceCordAIError):
    """Base exception for AI service-related errors."""

    pass


class AIConnectionError(AIServiceError):
    """Raised when AI service connection fails."""

    pass


class AIAuthenticationError(AIServiceError):
    """Raised when AI service authentication fails."""

    pass


class AIModelError(AIServiceError):
    """Raised when AI model operations fail."""

    pass


class DiscordBotError(VoiceCordAIError):
    """Base exception for Discord bot-related errors."""

    pass


class VoiceConnectionError(DiscordBotError):
    """Raised when Discord voice connection operations fail."""

    pass


class SessionError(DiscordBotError):
    """Raised when guild session operations fail."""

    pass


class ValidationError(VoiceCordAIError):
    """Raised when input validation fails."""

    pass


class StateTransitionError(VoiceCordAIError):
    """Raised when an invalid state transition is attempted."""

    pass
