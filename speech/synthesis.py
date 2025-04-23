"""
Speech synthesis module for converting text to emotionally expressive speech.
Supports multiple TTS engines with a focus on emotional expression.
"""
import logging
import os
import tempfile
from typing import Optional
import io

import elevenlabs
from elevenlabs.client import ElevenLabs
import boto3

import config

logger = logging.getLogger(__name__)

class SpeechSynthesizer:
    """
    Handles text-to-speech conversion with emotional expression capabilities.
    """
    
    def __init__(self):
        """Initialize the speech synthesizer with configured settings."""
        self.config = config.EMOTIONS
        self.audio_config = config.AUDIO
        
        # Set up ElevenLabs
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
        # Set up Amazon Polly as fallback
        self.polly_client = None
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            try:
                self.polly_client = boto3.client('polly')
                logger.info("Amazon Polly initialized as fallback TTS")
            except Exception as e:
                logger.error(f"Failed to initialize Amazon Polly: {e}")
        
        # Voice settings for different emotions
        self.voice_settings = {
            "neutral": {
                "stability": 0.75,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            },
            "happy": {
                "stability": 0.65,
                "similarity_boost": 0.75,
                "style": 0.3,
                "use_speaker_boost": True
            },
            "sad": {
                "stability": 0.85,
                "similarity_boost": 0.65,
                "style": 0.1,
                "use_speaker_boost": True
            },
            "angry": {
                "stability": 0.55,
                "similarity_boost": 0.85,
                "style": 0.4,
                "use_speaker_boost": True
            },
            "surprised": {
                "stability": 0.6,
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True
            },
            "concerned": {
                "stability": 0.8,
                "similarity_boost": 0.7,
                "style": 0.2,
                "use_speaker_boost": True
            }
        }
        
        logger.info("Speech synthesizer initialized")
    
    def speak(self, text: str, emotion: Optional[str] = None) -> bool:
        """
        Convert text to speech with the specified emotion and play it.
        
        Args:
            text: Text to convert to speech
            emotion: Emotion to express (or None for default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            logger.warning("Empty text provided to speech synthesizer")
            return False
        
        # Map agent state to emotion if needed
        if emotion in self.config["mapping"]:
            emotion = self.config["mapping"][emotion]
        
        # Use default emotion if not specified or invalid
        if not emotion or emotion not in self.config["available"]:
            emotion = self.config["default"]
        
        logger.info(f"Generating speech with emotion: {emotion}")
        
        try:
            # Get voice settings for the emotion
            voice_settings = self.voice_settings.get(
                emotion, 
                self.voice_settings[self.config["default"]]
            )
            
            # Generate speech with ElevenLabs using the correct method
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.config["voice_id"],
                model_id="eleven_multilingual_v2",
                voice_settings=voice_settings
            )
            
            # Collect all chunks from the generator into a bytes object
            audio_data = io.BytesIO()
            for chunk in audio_stream:
                audio_data.write(chunk)
            
            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio_data.getvalue())
                self._play_audio(temp_file.name)
                
            return True
            
        except Exception as e:
            logger.error(f"Error generating speech with ElevenLabs: {e}")
            
            # Try fallback to Amazon Polly if available
            if self.polly_client:
                return self._speak_with_polly(text, emotion)
            
            return False
    
    def _speak_with_polly(self, text: str, emotion: str) -> bool:
        """
        Fallback method to generate speech using Amazon Polly.
        
        Args:
            text: Text to convert to speech
            emotion: Emotion to express
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Map emotions to SSML with emotional markup
            ssml_text = self._add_emotional_ssml(text, emotion)
            
            response = self.polly_client.synthesize_speech(
                Engine='neural',
                OutputFormat='mp3',
                Text=ssml_text,
                TextType='ssml',
                VoiceId='Matthew'  # Can be configured based on preference
            )
            
            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(response['AudioStream'].read())
                self._play_audio(temp_file.name)
                
            return True
            
        except Exception as e:
            logger.error(f"Error generating speech with Amazon Polly: {e}")
            return False
    
    def _add_emotional_ssml(self, text: str, emotion: str) -> str:
        """
        Add SSML tags for emotional expression with Amazon Polly.
        
        Args:
            text: Text to convert
            emotion: Emotion to express
            
        Returns:
            str: SSML-formatted text
        """
        # Map emotions to Polly SSML
        emotion_mapping = {
            "happy": '<amazon:emotion name="excited" intensity="high">',
            "sad": '<amazon:emotion name="sad" intensity="medium">',
            "angry": '<amazon:emotion name="angry" intensity="medium">',
            "surprised": '<amazon:emotion name="excited" intensity="medium">',
            "concerned": '<amazon:emotion name="worried" intensity="medium">',
            "neutral": ''
        }
        
        opening_tag = emotion_mapping.get(emotion, '')
        closing_tag = '</amazon:emotion>' if opening_tag else ''
        
        return f'<speak>{opening_tag}{text}{closing_tag}</speak>'
    
    def _play_audio(self, file_path: str):
        """
        Play audio file using system's default audio player.
        
        Args:
            file_path: Path to audio file
        """
        try:
            # Use platform-specific commands to play audio
            import platform
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                os.system(f'afplay "{file_path}"')
            elif system == 'Linux':
                os.system(f'aplay "{file_path}"')
            elif system == 'Windows':
                os.system(f'start "{file_path}"')
            else:
                logger.error(f"Unsupported platform for audio playback: {system}")
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(file_path)
            except Exception:
                pass
    
    def close(self):
        """Clean up resources."""
        logger.info("Speech synthesizer closed")
