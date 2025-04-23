"""
Audio utility module for managing audio input and output.
Provides utilities for audio device management and processing.
"""
import logging
import pyaudio
import wave
import numpy as np
import tempfile
import os
from typing import Optional, List, Tuple, Dict

import config

logger = logging.getLogger(__name__)

class AudioManager:
    """
    Manages audio devices and provides utilities for audio processing.
    """
    
    def __init__(self):
        """Initialize the audio manager with configured settings."""
        self.config = config.AUDIO
        self.pyaudio = pyaudio.PyAudio()
        self.input_device = self._get_input_device()
        self.output_device = self._get_output_device()
        logger.info("Audio manager initialized")
    
    def _get_input_device(self) -> int:
        """
        Get the input device index.
        
        Returns:
            Input device index
        """
        if self.config["input_device"] is not None:
            return self.config["input_device"]
        
        # Use default input device
        return self.pyaudio.get_default_input_device_info()["index"]
    
    def _get_output_device(self) -> int:
        """
        Get the output device index.
        
        Returns:
            Output device index
        """
        if self.config["output_device"] is not None:
            return self.config["output_device"]
        
        # Use default output device
        return self.pyaudio.get_default_output_device_info()["index"]
    
    def list_devices(self) -> List[Dict]:
        """
        List available audio devices.
        
        Returns:
            List of audio device information
        """
        devices = []
        for i in range(self.pyaudio.get_device_count()):
            try:
                device_info = self.pyaudio.get_device_info_by_index(i)
                devices.append(device_info)
            except Exception as e:
                logger.error(f"Error getting device info for index {i}: {e}")
        
        return devices
    
    def record_audio(self, duration: float, sample_rate: int = 16000) -> np.ndarray:
        """
        Record audio for a specified duration.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Recorded audio as numpy array
        """
        frames = []
        chunk_size = 1024
        
        # Open stream
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=chunk_size
        )
        
        logger.info(f"Recording audio for {duration} seconds")
        
        # Record audio
        for _ in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)
        
        # Close stream
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        return audio_data
    
    def save_audio(self, audio_data: np.ndarray, file_path: str, sample_rate: int = 16000):
        """
        Save audio data to a file.
        
        Args:
            audio_data: Audio data as numpy array
            file_path: Path to save the audio file
            sample_rate: Sample rate in Hz
        """
        # Convert to int16
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Open wave file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        logger.info(f"Audio saved to {file_path}")
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Play audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
        """
        # Convert to int16
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            self.save_audio(audio_data, temp_file.name, sample_rate)
            temp_path = temp_file.name
        
        # Play the file
        self.play_audio_file(temp_path)
        
        # Clean up
        try:
            os.unlink(temp_path)
        except Exception:
            pass
    
    def play_audio_file(self, file_path: str):
        """
        Play an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        try:
            # Open the wave file
            with wave.open(file_path, 'rb') as wf:
                # Get file properties
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                
                # Create format based on sample width
                if sample_width == 1:
                    format = pyaudio.paInt8
                elif sample_width == 2:
                    format = pyaudio.paInt16
                elif sample_width == 3 or sample_width == 4:
                    format = pyaudio.paInt32
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # Open stream
                stream = self.pyaudio.open(
                    format=format,
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    output_device_index=self.output_device
                )
                
                # Read data
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                # Play audio
                logger.info(f"Playing audio file: {file_path}")
                while len(data) > 0:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
                
                # Close stream
                stream.stop_stream()
                stream.close()
                
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
    
    def close(self):
        """Clean up resources."""
        self.pyaudio.terminate()
        logger.info("Audio manager closed")
