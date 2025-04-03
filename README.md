# Discord Bot for Voice Chat Using GPT4o Realtime Model

This project is a Discord bot that allows users to have voice communication with gpt4o realtime api.

## Features

- **Voice Chat with GPT4o Realtime API**: Connects to a voice channel in Discord and can listen to audio from users in the channel.

## Setup

### Prerequisites

- **Python 3.8+** with `discord.py`, `websockets`, `voice_recv`, and `pydub` libraries installed
- A Discord bot token and OpenAI API key (stored in a `.env` file)

### Installation

0. install ffmpeg to your operating system, which is a dependency for pydub package.

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Monoese/VoiceCordAI
   ```
   
2. Create and activate a virtual environment for the project:
   if you are on unix
   ```plaintext
   python -m venv venv
   source venv/bin/activate
   ```
   if you are on windows
   ```plaintext
   py -m venv venv
   venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```plaintext
   pip install -r requirements.txt
   ```

4. Set up a `.env` file in the root directory with your tokens:
   ```plaintext
   DISCORD_TOKEN=your_discord_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Bot

To run the bot, execute:
   ```plaintext
   python main.py
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
