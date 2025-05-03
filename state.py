from enum import Enum
from typing import Optional

import discord
from discord import Message


class BotStateEnum(Enum):
    IDLE = "idle"
    STANDBY = "standby"
    RECORDING = "recording"


class BotState:
    def __init__(self):
        self._current_state: BotStateEnum = BotStateEnum.IDLE
        self._authority_user_id: Optional[str] = "anyone"
        self._authority_user_name: Optional[str] = "anyone"
        self._standby_message: Optional[Message] = None

    @property
    def current_state(self) -> BotStateEnum:
        return self._current_state

    @property
    def authority_user_id(self) -> str:
        return self._authority_user_id

    @authority_user_id.setter
    def authority_user_id(self, value: str):
        self._authority_user_id = value

    @property
    def standby_message(self) -> Optional[Message]:
        return self._standby_message

    def get_message_content(self) -> str:
        """Generate the standby message content based on current state."""
        return (f"**🎙 Voice Recording Bot - **{self._current_state.value}** Mode**\n\n"
                f"Here's how to control the bot:\n"
                f"---\n"
                f"### 🔄 How to Use:\n"
                f"1. **Start Recording**: React to this message with 🎙 to start recording.\n"
                f"2. **Stop Recording**: Remove your 🎙 reaction to pause recording.\n"
                f"4. **Finish Session**: Use `!-listen` to end the session and return the bot to Idle Mode.\n"
                f"---\n"
                f"### 🛠 Current Status:\n"
                f"- **Recording Status**: `{self._current_state.value}`\n"
                f"---\n"
                f"### 🧑 Authority User:\n"
                f"> `{self._authority_user_name}` can control the recording actions.")

    async def initialize_standby(self, ctx) -> bool:
        """Initialize standby state from idle."""
        if self._current_state != BotStateEnum.IDLE:
            return False

        self._current_state = BotStateEnum.STANDBY
        self._standby_message = await ctx.send(self.get_message_content())
        await self._standby_message.add_reaction("🎙")
        return True

    async def start_recording(self, user: discord.User) -> bool:
        """Transition to recording state."""
        if self._current_state != BotStateEnum.STANDBY:
            return False

        self._current_state = BotStateEnum.RECORDING
        self.authority_user_id = user.id
        self._authority_user_name = user.name
        await self._update_message()
        return True

    async def stop_recording(self) -> bool:
        """Stop recording and return to standby state."""
        if self._current_state != BotStateEnum.RECORDING:
            return False

        self._current_state = BotStateEnum.STANDBY
        self.authority_user_id = "anyone"
        self._authority_user_name = "anyone"
        await self._update_message()
        return True

    async def reset_to_idle(self) -> bool:
        """Reset the bot state to idle."""
        if self._current_state == BotStateEnum.IDLE:
            return False

        if self._standby_message:
            await self._standby_message.delete()
            self._standby_message = None

        self._current_state = BotStateEnum.IDLE
        self.authority_user_id = "anyone"
        return True

    async def _update_message(self):
        """Update the standby message with current state."""
        if self._standby_message:
            await self._standby_message.edit(content=self.get_message_content())

    def is_authorized(self, user: discord.User) -> bool:
        """Check if the user is authorized to control the bot."""
        return self.authority_user_id == "anyone" or user.id == self.authority_user_id
