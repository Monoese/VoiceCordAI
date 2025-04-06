# Discord Bot for Voice Chat Using GPT4o Realtime API

This project gives you a Discord bot that allows users to have voice communication with GPT-4o realtime api from OpenAI.

## Features

- **Voice Chat with GPT4o Realtime API**: The bot connects to a voice channel in Discord and have voice conversation with users empowered by GPT-4o Realtime API.

## Setup

### Prerequisites

- **Python 3.8+**
- A Discord bot token, which you can get after creating an application on discord developer portal, and OpenAI API key (both should be stored in a `.env` file which will be talked about later).

### Installation

0. install ffmpeg to your operating system, which is a dependency for pydub package. Make sure the "ffmpeg" command is accessible from your terminal.

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Monoese/VoiceCordAI
   cd <path_to_project_directory>
   ```

2. Create and activate a virtual environment for the project:
   if you are on unix
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   if you are on windows
   ```bash
   py -m venv venv
   venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up a `.env` file in the project root directory with your tokens:
   ```bash
   DISCORD_TOKEN=<your_discord_token_here>
   OPENAI_API_KEY=<your_openai_api_key_here>
   ```

### Running the Bot

To run the bot, execute:

   ```bash
   python main.py
   ```

## Commands

### Voice Channel Management

- **`!connect`** - Let bot join the user’s current voice channel.
- **`!disconnect`** - Let Bot leave the voice channel and disconnects from the WebSocket server.

### Recording Session Controls

- **`!listen`** - Initializes a recording session. Users can control the session by adding and removing reactions on a standby message.
- **`!-listen`** - Ends the recording session and sets the bot to idle mode.
- **Reactions**:
    - React with `🎙` - Starts recording audio in the session.
    - Remove react `🎙` - Stops recording audio in the session and send the audio to realtime API.
- **Playback** - The bot will start the playback of response from API as soon as a response websocket packet stream is getting received.

## Acknowledgments

- [GPT-4o Realtime API](https://platform.openai.com/docs/guides/realtime#connect-with-websockets) - OpenAI APIs for real-time interaction
- [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv) - Voice receive extension package
  for discord.py
- [discord.py](https://discordpy.readthedocs.io/) - Python wrapper for the Discord API.
- [pydub](https://github.com/jiaaro/pydub) - Audio processing library for Python.
