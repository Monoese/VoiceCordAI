# Discord Bot for Voice Chat Using GPT4o Realtime Model

This project is a Discord bot that allows users to control a voice recording session and handle interactions with a WebSocket server (such as OpenAI's real-time APIs). The bot offers various commands to manage voice recording sessions, capture and process audio, and handle real-time events.

## Features

- **Voice Channel Connection**: Connects to a voice channel in Discord and can listen to audio from users in the channel.
- **Audio Recording and Processing**: Captures audio in real-time, processes it, and encodes it to base64 for further use.
- **Interactive Controls**: Users can start, stop, and manage recordings through message reactions.
- **WebSocket Integration**: Sends and receives events through a WebSocket connection to a real-time server (like OpenAI’s APIs).
- **Event Queueing and Handling**: Manages event queues for both incoming and outgoing events to handle asynchronous tasks.
- **Bot Commands**: Provides several commands for connecting, starting sessions, and interacting with users.

## Setup

### Prerequisites

- **Python 3.8+** with `discord.py`, `websockets`, `voice_recv`, and `pydub` libraries installed
- A Discord bot token and OpenAI API key (stored in a `.env` file)

### Installation

1. Clone the repository and navigate to the project directory.
   ```bash
   git clone <your-repository-url>
   cd <project-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up a `.env` file in the root directory with your tokens:
   ```plaintext
   DISCORD_TOKEN=your_discord_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Bot

To run the bot, execute:
```bash
python bot.py
```

## Commands

### Voice Channel Management
- **`!connect`** - Bot joins the user’s current voice channel.
- **`!disconnect`** - Bot leaves the voice channel and disconnects from the WebSocket server.

### Recording Session Controls
- **`!listen`** - Initializes a recording session. Users can control the session using reactions on the standby message.
- **`!-listen`** - Ends the recording session and sets the bot to idle mode.
- **Reactions**:
  - React with `🎙` - Starts recording audio in the session.
  - Remove react `🎙` - Stops recording audio in the session and send the audio to realtime API.

## Acknowledgments

- OpenAI APIs for real-time interaction
- [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv) - Voice receive extension package for discord.py
- [discord.py](https://discordpy.readthedocs.io/) - Python wrapper for the Discord API.
- [pydub](https://github.com/jiaaro/pydub) - Audio processing library for Python.
