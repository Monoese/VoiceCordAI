import asyncio
import base64
from io import BytesIO

from discord import FFmpegPCMAudio
from discord.ext import voice_recv
from pydub import AudioSegment

from config import Config
from logger import get_logger

# Set up logger for this module
logger = get_logger(__name__)


class AudioManager:
    def __init__(self):
        self.response_buffer = bytearray()
        self.output_queue = asyncio.Queue()

    class PCM16Sink(voice_recv.AudioSink):
        def __init__(self):
            super().__init__()
            self.audio_data = bytearray()
            self.total_bytes = 0
            logger.debug("New audio sink initialized")

        def wants_opus(self) -> bool:
            return False

        def write(self, user, data):
            if data.pcm:
                self.audio_data.extend(data.pcm)
                self.total_bytes += len(data.pcm)
                logger.debug(f"Write called: Adding {len(data.pcm)} bytes. Total accumulated: {self.total_bytes} bytes")
            else:
                logger.warning("Write called with no PCM data")

        def cleanup(self):
            logger.debug(f"Cleanup called. Final total bytes: {self.total_bytes}")
            data_len = len(self.audio_data)
            logger.debug(f"Length of audio_data before clear: {data_len}")
            self.audio_data.clear()

    def create_sink(self):
        """Create a new sink instance"""
        return self.PCM16Sink()

    def extend_response_buffer(self, data):
        """Add new audio data to the response buffer"""
        self.response_buffer.extend(data)
        logger.debug(f"Added {len(data)} bytes to response buffer")

    def clear_response_buffer(self):
        """Clear the response buffer after processing"""
        self.response_buffer.clear()

    @staticmethod
    def process_audio(data):
        """Convert and process raw PCM data."""
        audio_segment = AudioSegment(data=data, sample_width=Config.SAMPLE_WIDTH, frame_rate=Config.DISCORD_FRAME_RATE,
                                     channels=Config.CHANNELS)

        audio_segment = audio_segment.set_frame_rate(Config.TARGET_FRAME_RATE)
        logger.debug(f"Processed audio sample rate: {audio_segment.frame_rate}")

        return audio_segment.raw_data

    @staticmethod
    def encode_to_base64(pcm_data):
        """Encode PCM data to base64 string."""
        return base64.b64encode(pcm_data).decode("utf-8")

    async def enqueue_audio(self, audio_buffer):
        """Process and enqueue audio for playback."""
        audio_segment = AudioSegment(data=bytes(audio_buffer), sample_width=Config.SAMPLE_WIDTH,
                                     frame_rate=Config.TARGET_FRAME_RATE, channels=Config.CHANNELS)

        audio_segment = audio_segment.set_frame_rate(Config.OUTPUT_FRAME_RATE).set_channels(Config.OUTPUT_CHANNELS)

        opus_buffer = BytesIO()
        audio_segment.export(opus_buffer, format="ogg", codec="libopus")
        opus_buffer.seek(0)

        await self.output_queue.put(opus_buffer)

    async def playback_loop(self, voice_client):
        """Handle audio playback loop."""
        while True:
            audio_buffer = await self.output_queue.get()

            try:
                audio_source = FFmpegPCMAudio(audio_buffer, pipe=True)
                # Create a callback function that uses the logger
                def log_playback_finished(error):
                    if error:
                        logger.error(f"Error during audio playback: {error}")
                    else:
                        logger.debug("Finished playing audio successfully")

                voice_client.play(audio_source, after=log_playback_finished)

                while voice_client.is_playing():
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error during audio playback: {e}")
            finally:
                self.output_queue.task_done()
