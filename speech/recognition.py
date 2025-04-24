"""
Speech recognition module for converting spoken language to text.
Supports multiple recognition engines including Whisper, Google, CMU Sphinx, and ElevenLabs.
"""
import logging
import queue
import threading
import time
import io
import re
import os
import tempfile
from typing import Optional, Callable

import numpy as np
import speech_recognition as sr
import whisper
from openai import OpenAI
from elevenlabs import ElevenLabs

import config

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    """
    Handles speech recognition using various engines with real-time streaming capabilities.
    """
    
    def __init__(self):
        """Initialize the speech recognizer with configured settings."""
        self.config = config.SPEECH_RECOGNITION
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Set up audio parameters
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 400  # Increased from 300 for better noise handling
        self.recognizer.pause_threshold = self.config["non_speaking_duration"]
        self.recognizer.phrase_threshold = self.config["phrase_threshold"]
        
        # For streaming recognition
        self.audio_queue = queue.Queue()
        self.stop_listening = None
        self.is_listening = False
        
        # Initialize clients for API-based speech recognition
        self.openai_client = None
        self.elevenlabs_client = None
        
        # Initialize Whisper model if selected
        self.whisper_model = None
        if self.config["engine"] == "whisper":
            try:
                logger.info("Loading Whisper model...")
                
                # Check which implementation to use
                if self.config.get("implementation") == "local":
                    # Use local Whisper model
                    model_size = self.config.get("model_size", "small")
                    logger.info(f"Using local Whisper model: {model_size}")
                    self.whisper_model = whisper.load_model(model_size)
                    
                elif self.config.get("implementation") == "openai_api":
                    # Use OpenAI API for Whisper
                    logger.info("Using Whisper via OpenAI API")
                    self.openai_client = OpenAI()
                    
                elif self.config.get("implementation") == "elevenlabs_api":
                    # Use ElevenLabs API for speech-to-text
                    logger.info("Using ElevenLabs speech-to-text API")
                    self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                    
                else:
                    # Default to local Whisper model
                    logger.info("Using default local Whisper model: small")
                    self.whisper_model = whisper.load_model("small")
                    
                logger.info("Speech recognition initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize speech recognition: {e}")
                logger.info("Falling back to Google Speech Recognition")
                self.config["engine"] = "google"
        
        # Calibrate microphone
        with self.microphone as source:
            logger.info("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased from 1 to 2 seconds
            logger.info("Microphone calibrated")
    
    def listen(self) -> Optional[str]:
        """
        Listen for speech and convert to text.
        
        Returns:
            str: Recognized text or None if recognition failed
        """
        try:
            with self.microphone as source:
                logger.info("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.config["timeout"],
                    phrase_time_limit=10
                )
            
            return self._recognize_audio(audio)
            
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout period")
            return None
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return None
    
    def start_streaming(self, callback: Callable[[str], None]):
        """
        Start streaming recognition with callback for continuous recognition.
        
        Args:
            callback: Function to call with recognized text
        """
        if self.is_listening:
            logger.warning("Already listening, ignoring start_streaming call")
            return
        
        self.is_listening = True
        
        def audio_processor():
            while self.is_listening:
                try:
                    audio = self.audio_queue.get(timeout=1)
                    text = self._recognize_audio(audio)
                    if text:
                        callback(text)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in audio processing thread: {e}")
        
        # Start processing thread
        threading.Thread(target=audio_processor, daemon=True).start()
        
        # Start listening in background
        self.stop_listening = self.recognizer.listen_in_background(
            self.microphone,
            self._audio_callback,
            phrase_time_limit=10
        )
        
        logger.info("Streaming recognition started")
    
    def stop_streaming(self):
        """Stop streaming recognition."""
        if not self.is_listening:
            return
            
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
            self.stop_listening = None
        
        self.is_listening = False
        logger.info("Streaming recognition stopped")
    
    def _audio_callback(self, recognizer, audio):
        """Callback for background listening to add audio to queue."""
        self.audio_queue.put(audio)
    
    def _recognize_audio(self, audio) -> Optional[str]:
        """
        Recognize speech from audio data using the configured engine.
        
        Args:
            audio: Audio data from recognizer
            
        Returns:
            str: Recognized text or None if recognition failed
        """
        try:
            if self.config["engine"] == "whisper":
                implementation = self.config.get("implementation", "local")
                
                # Option 1: ElevenLabs speech-to-text API
                if implementation == "elevenlabs_api" and self.elevenlabs_client:
                    try:
                        # Save audio to a temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_file_path = temp_file.name
                            temp_file.write(audio.get_wav_data())
                        
                        # Use ElevenLabs speech-to-text API
                        with open(temp_file_path, "rb") as audio_file:
                            audio_stream = io.BytesIO(audio_file.read())
                            result = self.elevenlabs_client.speech_to_text.convert(
                                file=audio_stream,
                                model_id="scribe_v1"
                            )
                        
                        os.unlink(temp_file_path)  # Clean up temp file
                        text = result.text
                        logger.info(f"ElevenLabs transcription: {text}")
                        return self._post_process_text(text)
                        
                    except Exception as e:
                        logger.error(f"Error with ElevenLabs API: {e}")
                        # Try to clean up temp file if it exists
                        try:
                            if 'temp_file_path' in locals():
                                os.unlink(temp_file_path)
                        except:
                            pass
                        
                        # Fall back to next option
                        logger.info("Falling back to OpenAI Whisper API")
                        implementation = "openai_api"
                
                # Option 2: OpenAI Whisper API
                if implementation == "openai_api" and self.openai_client:
                    try:
                        # Save audio to a temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_file_path = temp_file.name
                            temp_file.write(audio.get_wav_data())
                        
                        # Use OpenAI's Whisper API
                        with open(temp_file_path, "rb") as audio_file:
                            transcript = self.openai_client.audio.transcriptions.create(
                                model="whisper-1",  # This is the Whisper Turbo model via API
                                file=audio_file,
                                language=None if self.config["language"] == "auto" else self.config["language"],
                                temperature=0.0,
                                prompt="This is a conversation with an AI assistant."
                            )
                        
                        os.unlink(temp_file_path)  # Clean up temp file
                        text = transcript.text
                        logger.info(f"OpenAI Whisper API transcription: {text}")
                        return self._post_process_text(text)
                        
                    except Exception as e:
                        logger.error(f"Error with OpenAI Whisper API: {e}")
                        # Try to clean up temp file if it exists
                        try:
                            if 'temp_file_path' in locals():
                                os.unlink(temp_file_path)
                        except:
                            pass
                        
                        # Fall back to local model
                        logger.info("Falling back to local Whisper model")
                        implementation = "local"
                
                # Option 3: Local Whisper model
                if implementation == "local" or (implementation not in ["elevenlabs_api", "openai_api"]):
                    if not self.whisper_model:
                        # Try to load the model if it's not already loaded
                        try:
                            model_size = self.config.get("model_size", "small")
                            self.whisper_model = whisper.load_model(model_size)
                            logger.info(f"Loaded local Whisper model: {model_size}")
                        except Exception as e:
                            logger.error(f"Failed to load local Whisper model: {e}")
                            return None
                    
                    # Convert audio to numpy array for Whisper
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Use local Whisper model
                    result = self.whisper_model.transcribe(
                        audio_data, 
                        language=None if self.config["language"] == "auto" else self.config["language"],
                        fp16=False,
                        temperature=0.0,
                        initial_prompt="The following is a conversation with an AI assistant."
                    )
                    
                    text = result["text"].strip()
                    logger.info(f"Local Whisper transcription: {text}")
                    return self._post_process_text(text)
            
            elif self.config["engine"] == "google":
                text = self.recognizer.recognize_google(
                    audio,
                    language=self.config["language"] if self.config["language"] != "auto" else None
                )
                return self._post_process_text(text)
                
            elif self.config["engine"] == "sphinx":
                text = self.recognizer.recognize_sphinx(audio)
                return self._post_process_text(text)
                
            else:
                logger.error(f"Unsupported recognition engine: {self.config['engine']}")
                return None
                
        except sr.UnknownValueError:
            logger.info("Speech not understood")
            return None
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return None
    
    def _post_process_text(self, text: str) -> str:
        """
        Apply corrections to common transcription errors.
        
        Args:
            text: Raw transcription text
            
        Returns:
            str: Processed and corrected text
        """
        if not text:
            return text
            
        # Common corrections
        corrections = {
            "i'm": "I'm",
            "im ": "I'm ",
            "dont": "don't",
            "cant": "can't",
            "wont": "won't",
            "ive": "I've",
            "id ": "I'd ",
            "ill ": "I'll ",
            "youre": "you're",
            "theyre": "they're",
            "isnt": "isn't",
            "didnt": "didn't",
            "thats": "that's",
            "whats": "what's",
            "lets": "let's",
            "its ": "it's ",
            "wasnt": "wasn't",
            "wouldnt": "wouldn't",
            "couldnt": "couldn't",
            "shouldnt": "shouldn't",
            "weve": "we've",
            "theyve": "they've",
            "youve": "you've",
            "havent": "haven't",
        }
        
        # Apply corrections
        for incorrect, correct in corrections.items():
            text = re.sub(r'\b' + incorrect + r'\b', correct, text, flags=re.IGNORECASE)
        
        # Capitalize first letter of sentences
        text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.!?,;:])([a-zA-Z])', r'\1 \2', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def close(self):
        """Clean up resources."""
        self.stop_streaming()
        logger.info("Speech recognizer closed")
