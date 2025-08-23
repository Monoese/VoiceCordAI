"""
Unit tests for the unified audio processing framework.

Tests cover all aspects of the new processing module including:
- AudioFormat functionality
- RealtimeAudioProcessor (audioop-based)
- QualityAudioProcessor (pydub-based)
- UnifiedAudioProcessor facade and auto-selection
- Utility functions for common operations
"""

import base64
import io
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch


from src.audio.processing import (
    AudioFormat,
    ProcessingStrategy,
    RealtimeAudioProcessor,
    QualityAudioProcessor,
    UnifiedAudioProcessor,
    DISCORD_FORMAT,
    WAKE_WORD_FORMAT,
    VAD_FORMAT,
    OPENAI_FORMAT,
    load_audio_file,
    encode_pcm_to_base64,
    decode_base64_to_pcm,
    encode_pcm_to_base64_async,
    decode_base64_to_pcm_async
)
from src.exceptions import AudioProcessingError


class TestAudioFormat:
    """Test AudioFormat dataclass and predefined formats."""
    
    def test_audio_format_creation(self):
        """Test AudioFormat creation with various parameters."""
        format1 = AudioFormat(48000, 2, 2)
        assert format1.sample_rate == 48000
        assert format1.channels == 2
        assert format1.sample_width == 2
        
        # Test default sample_width
        format2 = AudioFormat(16000, 1)
        assert format2.sample_width == 2
    
    def test_audio_format_properties(self):
        """Test AudioFormat properties."""
        mono_format = AudioFormat(16000, 1, 2)
        assert mono_format.is_mono is True
        assert mono_format.is_stereo is False
        
        stereo_format = AudioFormat(48000, 2, 2)
        assert stereo_format.is_mono is False
        assert stereo_format.is_stereo is True
    
    def test_audio_format_string_representation(self):
        """Test AudioFormat string representation."""
        format_obj = AudioFormat(48000, 2, 2)
        expected = "48000Hz, 2ch, 16bit"
        assert str(format_obj) == expected
    
    def test_audio_format_equality(self):
        """Test AudioFormat equality comparison."""
        format1 = AudioFormat(48000, 2, 2)
        format2 = AudioFormat(48000, 2, 2)
        format3 = AudioFormat(16000, 1, 2)
        
        assert format1 == format2
        assert format1 != format3
    
    def test_predefined_formats(self):
        """Test that predefined formats have expected values."""
        # Test that formats are properly defined (not testing specific values 
        # since those come from Config, just structure)
        assert isinstance(DISCORD_FORMAT, AudioFormat)
        assert isinstance(WAKE_WORD_FORMAT, AudioFormat)
        assert isinstance(VAD_FORMAT, AudioFormat)
        assert isinstance(OPENAI_FORMAT, AudioFormat)
        
        # Test that wake word and VAD are mono
        assert WAKE_WORD_FORMAT.is_mono
        assert VAD_FORMAT.is_mono
        assert OPENAI_FORMAT.is_mono


class TestRealtimeAudioProcessor:
    """Test RealtimeAudioProcessor with audioop backend."""
    
    @pytest.fixture
    def processor(self):
        """Create a RealtimeAudioProcessor for testing."""
        return RealtimeAudioProcessor()
    
    @pytest.fixture
    def sample_audio_stereo(self):
        """Create sample stereo PCM data for testing."""
        # Simple sine wave in stereo (16-bit, 48kHz)
        import math
        sample_rate = 48000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        audio_data = bytearray()
        for i in range(samples):
            # Generate sine wave
            value = int(16384 * math.sin(2 * math.pi * 440 * i / sample_rate))  # 440Hz tone
            # Add as 16-bit little-endian stereo (left and right channels same)
            audio_data.extend(value.to_bytes(2, 'little', signed=True))  # Left
            audio_data.extend(value.to_bytes(2, 'little', signed=True))  # Right
        
        return bytes(audio_data)
    
    @pytest.fixture
    def sample_audio_mono(self):
        """Create sample mono PCM data for testing."""
        import math
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        audio_data = bytearray()
        for i in range(samples):
            value = int(16384 * math.sin(2 * math.pi * 440 * i / sample_rate))
            audio_data.extend(value.to_bytes(2, 'little', signed=True))
        
        return bytes(audio_data)
    
    def test_processor_initialization(self, processor):
        """Test processor initializes with empty state."""
        assert len(processor._resample_states) == 0
    
    def test_same_format_conversion(self, processor, sample_audio_mono):
        """Test conversion when source and target formats are identical."""
        format_obj = AudioFormat(16000, 1, 2)
        
        result = processor.convert_sync(
            sample_audio_mono, format_obj, format_obj
        )
        
        # Should return original data unchanged
        assert result == sample_audio_mono
    
    def test_empty_data_handling(self, processor):
        """Test handling of empty audio data."""
        source_format = DISCORD_FORMAT
        target_format = WAKE_WORD_FORMAT
        
        result = processor.convert_sync(b"", source_format, target_format)
        assert result == b""
    
    @pytest.mark.asyncio
    async def test_async_conversion(self, processor, sample_audio_stereo):
        """Test asynchronous conversion."""
        result = await processor.convert_async(
            sample_audio_stereo,
            DISCORD_FORMAT,
            WAKE_WORD_FORMAT,
            state_key="test_async"
        )
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Should be shorter (mono) than original (stereo)
        assert len(result) < len(sample_audio_stereo)
    
    def test_stateful_processing(self, processor, sample_audio_stereo):
        """Test that stateful processing maintains state between chunks."""
        state_key = "test_state"
        
        # Process first chunk
        result1 = processor.convert_sync(
            sample_audio_stereo[:1000],
            DISCORD_FORMAT,
            WAKE_WORD_FORMAT,
            state_key=state_key
        )
        
        # Check that state was created
        assert state_key in processor._resample_states
        first_state = processor._resample_states[state_key]
        
        # Process second chunk
        result2 = processor.convert_sync(
            sample_audio_stereo[1000:2000],
            DISCORD_FORMAT,
            WAKE_WORD_FORMAT,
            state_key=state_key
        )
        
        # State should have been updated
        second_state = processor._resample_states[state_key]
        assert second_state != first_state  # State should change
        
        assert len(result1) > 0
        assert len(result2) > 0
    
    def test_state_management(self, processor):
        """Test state reset and clearing functionality."""
        # Create some state
        processor._resample_states["test1"] = "dummy_state"
        processor._resample_states["test2"] = "dummy_state"
        
        # Reset specific state
        processor.reset_state("test1")
        assert "test1" not in processor._resample_states
        assert "test2" in processor._resample_states
        
        # Clear all states
        processor.clear_all_states()
        assert len(processor._resample_states) == 0
    
    
    def test_invalid_audio_data_handling(self, processor):
        """Test handling of invalid audio data."""
        with pytest.raises(AudioProcessingError):
            processor.convert_sync(
                b"invalid_audio_data",
                DISCORD_FORMAT,
                WAKE_WORD_FORMAT
            )


class TestQualityAudioProcessor:
    """Test QualityAudioProcessor with pydub backend."""
    
    @pytest.fixture
    def processor(self):
        """Create a QualityAudioProcessor for testing."""
        return QualityAudioProcessor()
    
    @pytest.fixture
    def sample_audio_segment(self):
        """Create sample audio using pydub for testing."""
        # Generate 100ms of 440Hz tone
        from pydub.generators import Sine
        tone = Sine(440).to_audio_segment(duration=100)  # 100ms
        return tone.set_frame_rate(48000).set_channels(2)  # Discord format
    
    @pytest.fixture
    def sample_audio_bytes(self, sample_audio_segment):
        """Get raw bytes from sample audio segment."""
        buffer = io.BytesIO()
        sample_audio_segment.export(buffer, format="raw")
        return buffer.getvalue()
    
    def test_processor_initialization(self, processor):
        """Test processor initializes correctly."""
        # Just verify the processor exists
        assert processor is not None
    
    def test_same_format_conversion(self, processor, sample_audio_bytes):
        """Test conversion when formats are identical."""
        format_obj = DISCORD_FORMAT
        
        result = processor.convert_sync(
            sample_audio_bytes, format_obj, format_obj
        )
        
        assert result == sample_audio_bytes
    
    def test_empty_data_handling(self, processor):
        """Test handling of empty audio data."""
        result = processor.convert_sync(b"", DISCORD_FORMAT, WAKE_WORD_FORMAT)
        assert result == b""
    
    @pytest.mark.asyncio
    async def test_async_conversion(self, processor, sample_audio_bytes):
        """Test asynchronous conversion."""
        result = await processor.convert_async(
            sample_audio_bytes,
            DISCORD_FORMAT,
            WAKE_WORD_FORMAT
        )
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Should be smaller (mono, lower sample rate)
        assert len(result) < len(sample_audio_bytes)
    
    def test_sample_rate_conversion(self, processor, sample_audio_bytes):
        """Test sample rate conversion."""
        source_format = DISCORD_FORMAT  # 48kHz
        target_format = WAKE_WORD_FORMAT  # 16kHz
        
        result = processor.convert_sync(
            sample_audio_bytes, source_format, target_format
        )
        
        # Should be approximately 1/3 the size due to 48k->16k conversion
        # and also half due to stereo->mono
        expected_ratio = (16000 / 48000) * (1 / 2)  # sample rate * channel reduction
        actual_ratio = len(result) / len(sample_audio_bytes)
        
        # Allow some tolerance for pydub processing differences
        assert abs(actual_ratio - expected_ratio) < 0.1
    
    
    def test_reset_state_noop(self, processor):
        """Test that reset_state is a no-op for stateless processor."""
        # Should not raise an exception
        processor.reset_state("any_key")
        # No way to verify internal state since it's stateless


class TestUnifiedAudioProcessor:
    """Test UnifiedAudioProcessor facade and strategy selection."""
    
    @pytest.fixture
    def processor(self):
        """Create a UnifiedAudioProcessor for testing."""
        return UnifiedAudioProcessor()
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        # Simple 16-bit stereo PCM data
        return b"sample_audio_data" * 100  # Make it reasonable size
    
    def test_processor_initialization(self, processor):
        """Test processor initializes with both backends."""
        assert processor._realtime_processor is not None
        assert processor._quality_processor is not None
    
    def test_explicit_strategy_selection(self, processor):
        """Test explicit strategy selection."""
        sample_data = b"test_data"
        
        # Test REALTIME strategy selection
        strategy = processor._select_strategy(
            ProcessingStrategy.REALTIME, sample_data, None
        )
        assert strategy == ProcessingStrategy.REALTIME
        
        # Test QUALITY strategy selection
        strategy = processor._select_strategy(
            ProcessingStrategy.QUALITY, sample_data, None
        )
        assert strategy == ProcessingStrategy.QUALITY
    
    def test_auto_strategy_selection_with_state_key(self, processor):
        """Test AUTO strategy selects REALTIME when state_key is provided."""
        large_data = b"x" * 200000  # 200KB
        
        strategy = processor._select_strategy(
            ProcessingStrategy.AUTO, large_data, "some_state_key"
        )
        assert strategy == ProcessingStrategy.REALTIME
    
    def test_auto_strategy_selection_small_data(self, processor):
        """Test AUTO strategy selects REALTIME for small data."""
        small_data = b"x" * 1000  # 1KB
        
        strategy = processor._select_strategy(
            ProcessingStrategy.AUTO, small_data, None
        )
        assert strategy == ProcessingStrategy.REALTIME
    
    def test_auto_strategy_selection_large_data(self, processor):
        """Test AUTO strategy selects QUALITY for large data without state_key."""
        large_data = b"x" * 200000  # 200KB
        
        strategy = processor._select_strategy(
            ProcessingStrategy.AUTO, large_data, None
        )
        assert strategy == ProcessingStrategy.QUALITY
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, processor):
        """Test handling of empty data."""
        result = await processor.convert(
            DISCORD_FORMAT, WAKE_WORD_FORMAT, b""
        )
        assert result == b""
    
    @pytest.mark.asyncio
    async def test_convert_with_realtime_strategy(self, processor, sample_audio):
        """Test conversion using REALTIME strategy."""
        with patch.object(processor._realtime_processor, 'convert_async') as mock_convert:
            mock_convert.return_value = b"realtime_result"
            
            result = await processor.convert(
                DISCORD_FORMAT,
                WAKE_WORD_FORMAT,
                sample_audio,
                strategy=ProcessingStrategy.REALTIME,
                state_key="test_state"
            )
            
            assert result == b"realtime_result"
            mock_convert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_convert_with_quality_strategy(self, processor, sample_audio):
        """Test conversion using QUALITY strategy."""
        with patch.object(processor._quality_processor, 'convert_async') as mock_convert:
            mock_convert.return_value = b"quality_result"
            
            result = await processor.convert(
                DISCORD_FORMAT,
                WAKE_WORD_FORMAT,
                sample_audio,
                strategy=ProcessingStrategy.QUALITY
            )
            
            assert result == b"quality_result"
            mock_convert.assert_called_once()
    
    def test_state_management(self, processor):
        """Test state management delegation to realtime processor."""
        with patch.object(processor._realtime_processor, 'reset_state') as mock_reset:
            processor.reset_state("test_key")
            mock_reset.assert_called_once_with("test_key")
        
        with patch.object(processor._realtime_processor, 'clear_all_states') as mock_clear:
            processor.clear_all_states()
            mock_clear.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_encode_pcm_to_base64(self):
        """Test PCM to base64 encoding."""
        test_data = b"test_pcm_data"
        encoded = encode_pcm_to_base64(test_data)
        
        assert isinstance(encoded, str)
        # Should be able to decode back to original
        decoded = base64.b64decode(encoded)
        assert decoded == test_data
    
    def test_decode_base64_to_pcm(self):
        """Test base64 to PCM decoding."""
        test_data = b"test_pcm_data"
        encoded = base64.b64encode(test_data).decode('utf-8')
        
        decoded = decode_base64_to_pcm(encoded)
        assert decoded == test_data
    
    @pytest.mark.asyncio
    async def test_load_audio_file(self):
        """Test loading audio file functionality."""
        # Create a temporary WAV file for testing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            # Create a simple audio segment and save it
            from pydub.generators import Sine
            tone = Sine(440).to_audio_segment(duration=100)  # 100ms 440Hz tone
            tone.export(tmp_file_path, format="wav")
            
        try:
            # Test loading the file (after the context manager closes the file handle)
            pcm_data, format_info = await load_audio_file(tmp_file_path)
            
            assert isinstance(pcm_data, bytes)
            assert len(pcm_data) > 0
            assert isinstance(format_info, AudioFormat)
            assert format_info.sample_rate > 0
            assert format_info.channels > 0
            
        finally:
            # Clean up
            Path(tmp_file_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(AudioProcessingError):
            await load_audio_file("nonexistent_file.wav")


class TestAsyncBase64Functions:
    """Test async base64 encoding/decoding functions for AI services."""
    
    @pytest.mark.asyncio
    async def test_encode_pcm_to_base64_async(self):
        """Test async PCM to base64 encoding."""
        test_data = b"test_pcm_data_async"
        encoded = await encode_pcm_to_base64_async(test_data)
        
        assert isinstance(encoded, str)
        # Should be able to decode back to original
        decoded = base64.b64decode(encoded)
        assert decoded == test_data
    
    @pytest.mark.asyncio
    async def test_decode_base64_to_pcm_async(self):
        """Test async base64 to PCM decoding."""
        test_data = b"test_pcm_data_async"
        encoded = base64.b64encode(test_data).decode('utf-8')
        
        decoded = await decode_base64_to_pcm_async(encoded)
        assert decoded == test_data
    
    @pytest.mark.asyncio
    async def test_async_base64_roundtrip(self):
        """Test complete async encode/decode roundtrip."""
        original_data = b"complex_pcm_audio_data" * 100
        
        # Encode async
        encoded = await encode_pcm_to_base64_async(original_data)
        
        # Decode async 
        decoded = await decode_base64_to_pcm_async(encoded)
        
        assert decoded == original_data