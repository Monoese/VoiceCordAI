# VoiceCordAI - Discord Voice Chat Bot

A Discord bot that enables real-time voice conversations between users and OpenAI's GPT-4 API. Users can engage in natural voice interactions with the AI through Discord voice channels.

## Features

- Real-time voice interaction with GPT-4
- Seamless Discord voice channel integration
- Simple reaction-based controls
- Error handling and logging

## Prerequisites

Before you start, make sure you have:
- Python 3.8 or newer installed (otherwise you can't use discord-ext-voice-recv library)
- FFmpeg installed (on windows, you can get it installed using winget by running: winget install -e --id Gyan.FFmpeg)
- A Discord Bot Token (from Discord Developer Portal)
- An OpenAI API Key (from OpenAI dashboard: platform.openai.com/api-keys)

## Step-by-Step Installation Guide

1. **Download the Project**
   - Open your terminal or command prompt
   - Copy and paste these commands if you are in your user's home directory:
   ```bash
   git clone https://github.com/Monoese/VoiceCordAI
   cd VoiceCordAI
   ```

2. **Create a Virtual Environment for the Bot**
   - This keeps the bot's requirements separate from other Python projects
   - First, create the virtual environment:
   
   Windows:
   ```bash
   py -m venv .venv
   ```
   
   macOS/Linux:
   ```bash
   python -m venv .venv
   ```
   
   - Then activate it based on your operating system:
   
   Windows:
   ```bash
   .\.venv\Scripts\activate
   ```
   
   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
   
   Note: When the environment is activated, you'll see `(venv)` at the start of your command line

3. **Install Required Programs**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Your Bot Settings**
   - Create a new file named `.env` in the main folder
   - Open it with any text editor
   - Add these lines (replace with your actual tokens):
   ```
   DISCORD_TOKEN=your_discord_token
   OPENAI_API_KEY=your_openai_api_key
   ```

## Starting the Bot

1. **Launch the Bot**
   - Make sure you're in the project folder
   - Make sure your virtual environment is activated (you see `(venv)`)
   - Run the bot using:
   
   Windows:
   ```bash
   py -m src.bot.main
   ```
   
   macOS/Linux:
   ```bash
   python -m src.bot.main
   ```

2. **Available Commands**
   - `/connect` - Joins your current voice channel, connects to the AI service, and prepares for voice interaction (standby mode).
   - `/disconnect` - Leaves the voice channel, disconnects from the AI service, and resets the bot to an idle state.

3. **Voice Controls**
   - Click 🎙️ to start recording
   - Click 🎙️ again to stop recording and process audio
   - Click ❌ to stop recording

## Need Help?
If you run into any issues, make sure:
1. Your virtual environment is activated (if you see `(venv)`)
2. All requirements are installed
3. Your `.env` file contains valid tokens
4. You're in the correct directory when running commands

## Acknowledgments

- [OpenAI GPT-4 Realtime API](https://platform.openai.com/docs/guides/realtime)
- [discord.py](https://discordpy.readthedocs.io/)
- [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv)
