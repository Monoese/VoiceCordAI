# VoiceCordAI - Discord Voice Chat Bot

A Discord bot that enables real-time voice conversations between users and various AI services, including OpenAI's GPT-4 and Google's Gemini. Users can engage in natural voice interactions with the selected AI through Discord voice channels.

## Features

- Talk with AI (OpenAI GPT-4, Google Gemini) in real-time using your voice.
- Switch between different AI service providers.
- Works directly in Discord voice channels.
- Easy to control using emoji reactions, which mimics a push-to-talk button.
- Includes error handling and activity logging.

## Getting Started

Follow these steps to set up and run the bot on your local machine or a virtual machine (VM).

### 1. Prerequisites

Before you begin, ensure you have the following software installed:

- **Python 3.8 or newer**
- **FFmpeg** (e.g., `winget install -e --id Gyan.FFmpeg` on Windows)

You will also need to gather API keys and set up your Discord bot.

#### Setting Up API Keys and Discord Bot

1.  **Discord Bot Token:**
    - Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a "New Application".
    - Go to the "Installation" tab and set the "Install Link" to "None".
    - Navigate to the "Bot" tab, in "Token" section, click "Reset Token" to obtain a private token and keep it secured.
    - Within "Bot" tab, Disable "Public Bot" option so that only you can join this bot to servers.
    - Again, within "Bot" tab, enable "Message Content Intent".
    - **Invite the bot to your server:** 
      - Go to the "OAuth2" tab and go to the "OAuth2 URL Generator" section.
      - Select `bot` as the scope and ensure `Guild Install` is the selected integration type.
      - In the "Bot Permissions" section, grant the following permissions:
        - `View Channels`
        - `Send Messages`
        - `Add Reactions`
        - `Connect`
        - `Speak`
      - Make sure you have sufficient permission to invite bot to the server you want to use the bot within.
      - Copy the generated URL and paste it into your browser to add the bot to your server.

2.  **OpenAI API Key:**
    - Obtain your key from the [OpenAI API keys](https://platform.openai.com/api-keys) page.

3.  **Google Gemini API Key:**
    - Obtain your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

*Note: The bot requires at least one API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`) to function. You do not have to provide both.*

### 2. Installation and Configuration

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Monoese/VoiceCordAI
    cd VoiceCordAI
    ```

2.  **Set Up a Virtual Environment**
    Create and activate a virtual environment in the project directory.

    **On Windows:**
    ```bash
    # Create the environment
    py -m venv .venv
    # Activate it
    .\.venv\Scripts\activate
    ```

    **On macOS/Linux:**
    ```bash
    # Create the environment
    python -m venv .venv
    # Activate it
    source .venv/bin/activate
    ```
    *You should see `(.venv)` at the beginning of your command prompt.*

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys**
    - Create a file named `.env` in the project's root directory.
    - Add your credentials to it like this:
      ```
      DISCORD_TOKEN=your_discord_token
      OPENAI_API_KEY=your_openai_api_key
      GEMINI_API_KEY=your_gemini_api_key
      ```
    *Note: The bot requires at least one API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`) to function. You do not need to provide both.*

### 3. Running the Bot

This project can be run locally or on a server/VM. With your virtual environment still activated, start the bot.

**On Windows:**
```bash
py -m src.bot.main
```

**On macOS/Linux:**
```bash
python -m src.bot.main
```

## Usage

### Bot Commands

| Command | Description | Example |
|---|---|---|
| `/connect` | Joins your voice channel and enters standby mode. | `/connect` |
| `/disconnect` | Leaves the voice channel and resets the bot. | `/disconnect` |
| `/set` | Sets the AI provider (`openai` or `gemini`). OpenAI is recommended. | `/set openai` |

### Voice Controls

After using `/connect`, the bot posts a status message. Use reactions on that message to control it:

- **🎙️ Start Recording:** React to begin capturing your voice.
- **🎙️ Stop Recording:** React again to stop and send your audio to the AI.
- **❌ Cancel:** React to stop and discard the current recording.

## Troubleshooting

If you encounter issues, please verify that:
- Your virtual environment is active (you see `(.venv)` in your terminal).
- All dependencies from `requirements.txt` are installed.
- The `.env` file exists and contains valid API keys.
- The bot was invited to your server with the correct permissions (see Step 1).
- You are in the project's root directory when running the bot.

## Acknowledgments

- [OpenAI API](https://platform.openai.com/docs/guides/realtime)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs/live)
- [discord.py](https://discordpy.readthedocs.io/)
- [discord-ext-voice-recv](https://github.com/imayhaveborkedit/discord-ext-voice-recv)
