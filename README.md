# VoiceCordAI - Discord Voice Chat Bot

A Discord bot that enables real-time voice conversations between users and OpenAI's GPT-4 API. Users can engage in natural voice interactions with the AI through Discord voice channels.

## Features

- Real-time voice interaction with GPT-4
- Seamless Discord voice channel integration
- Simple reaction-based controls
- Automatic audio processing and conversion
- Robust error handling and logging
- Configurable audio settings

## Prerequisites

- Python 3.8 or higher
- FFmpeg (accessible from terminal)
- Discord Bot Token
- OpenAI API Key

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Monoese/VoiceCordAI
   cd VoiceCordAI
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Unix/macOS
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the project root:
   ```
   DISCORD_TOKEN=your_discord_token
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. **Start the Bot**
   ```bash
   python main.py
   ```

2. **Available Commands**
   - `!connect` - Join your current voice channel
   - `!disconnect` - Leave the voice channel
   - `!listen` - Start a listening session
   - `!-listen` - End the listening session

3. **Voice Interaction Controls**
   - React with 🎙️ to start recording
   - Remove 🎙️ reaction to stop recording and process audio
   - React with ❌ to cancel current recording

## Acknowledgments

- [OpenAI GPT-4 Realtime API](https://platform.openai.com/docs/guides/realtime)
- [discord.py](https://discordpy.readthedocs.io/)
- [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv)
