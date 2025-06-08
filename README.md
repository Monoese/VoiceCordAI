# VoiceCordAI - Discord Voice Chat Bot

A Discord bot that enables real-time voice conversations between users and various AI services, including OpenAI's GPT-4 and Google's Gemini. Users can engage in natural voice interactions with the selected AI through Discord voice channels.

## Features

- Talk with AI (OpenAI GPT-4, Google Gemini) in real-time using your voice
- Switch between different AI service providers
- Works directly in Discord voice channels
- Easy to control using emoji reactions
- Includes error handling and activity logging

## Prerequisites

Before you start, make sure you have:
- Python 3.8 or newer installed (otherwise you can't use discord-ext-voice-recv library)
- FFmpeg installed (on windows, you can get it installed using winget by running: winget install -e --id Gyan.FFmpeg)
- A Discord Bot Token (from Discord Developer Portal)
- An OpenAI API Key (from OpenAI dashboard: platform.openai.com/api-keys)
- A Google Gemini API Key (from Google AI Studio: aistudio.google.com/app/apikey)
- Invited your bot to your Discord server with permissions to: Connect, Speak, Read Messages/View Channels, and Use Voice Activity.

## Step-by-Step Installation Guide

1. **Download the Project**
   - Open your terminal or command prompt
   - Run the following commands to download the project and navigate into its main folder:
   ```bash
   git clone https://github.com/Monoese/VoiceCordAI
   cd VoiceCordAI
   ```

2. **Create a Virtual Environment for the Bot**
   - This keeps the bot's requirements separate from other Python projects
   
   **Windows:**
   - Create the virtual environment:
   ```bash
   py -m venv .venv
   ```
   - Activate it:
   ```bash
   .\.venv\Scripts\activate
   ```
   
   ---
   
   **macOS/Linux:**
   - Create the virtual environment:
   ```bash
   python -m venv .venv
   ```
   - Activate it:
   ```bash
   source .venv/bin/activate
   ```
   
   Note: When the environment is activated, you'll see `(.venv)` at the start of your command line

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
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Starting the Bot

1. **Launch the Bot**
   - Make sure you're in the project folder
   - Make sure your virtual environment is activated (you see `(.venv)`)
   - Run the bot using:
   
   **Windows:**
   ```bash
   py -m src.bot.main
   ```
   
   **macOS/Linux:**
   ```bash
   python -m src.bot.main
   ```

2. **Available Commands**
   - `/connect` - Joins your current voice channel, connects to the currently selected AI service, and prepares for voice interaction (standby mode).
   - `/set <service_provider>` - Sets the AI service provider to use. Options: `openai` or `gemini`. Example: `/set openai`
   - `/disconnect` - Leaves the voice channel, disconnects from the AI service, and resets the bot to an idle state.

3. **Voice Controls**
   After using `/connect`, the bot will send a message indicating the active AI provider. React to this message to control recording:
   - React with 🎙️ to start recording your voice.
   - React with 🎙️ again to stop recording and send your audio for processing.
   - React with ❌ to cancel the current recording without sending it.

## Need Help?
If you run into any issues, make sure:
1. Your virtual environment is activated (you should see `(.venv)` at the start of your command line).
2. All requirements are installed.
3. Your `.env` file contains valid tokens.
4. You're in the correct directory when running commands

## Acknowledgments

- [OpenAI API](https://platform.openai.com/docs/guides/gpt)
- [Google Gemini API](https://ai.google.dev/docs/gemini_api_overview)
- [discord.py](https://discordpy.readthedocs.io/)
- [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv)
