"""
Configuration settings for the voicebot system.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Speech Recognition Settings
SPEECH_RECOGNITION = {
    "engine": "whisper",  # Options: "whisper", "google", "sphinx"
    "implementation": "elevenlabs_api",  # Options: "local", "openai_api", "elevenlabs_api"
    "model_size": "turbo",  # Options for local model: "tiny", "base", "small", "medium", "large", "turbo"
    "language": "en-US",
    "sample_rate": 16000,
    "chunk_size": 1024,
    "timeout": 5,  # seconds
    "phrase_threshold": 0.3,
    "non_speaking_duration": 1.0,
}

# NLU Settings
NLU = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 150,
    "context_window": 10,  # Number of conversation turns to remember
}

# Agent Settings
AGENT = {
    "reasoning_depth": 3,  # How many reasoning steps to take
    "confidence_threshold": 0.7,  # Minimum confidence to proceed without clarification
    "max_clarification_turns": 2,  # Maximum number of times to ask for clarification
}

# MCP Settings
MCP = {
    "endpoint": "ws://localhost:8765",
    "timeout": 30,  # seconds
    "retry_attempts": 3,
    "retry_delay": 2,  # seconds
}

# Emotional Speech Settings
EMOTIONS = {
    "default": "neutral",
    "available": ["neutral", "happy", "sad", "angry", "surprised", "concerned"],
    "mapping": {
        # Maps agent states to emotions
        "greeting": "happy",
        "information": "neutral",
        "error": "concerned",
        "success": "happy",
        "warning": "concerned",
        "apology": "sad",
    },
    "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Default ElevenLabs voice ID
    "stability": 0.75,
    "similarity_boost": 0.75,
    "style": 0.0,  # How much to apply speaking style (0-1)
    "use_speaker_boost": True,
}

# Audio Settings
AUDIO = {
    "output_device": None,  # None for default
    "input_device": None,  # None for default
    "volume": 1.0,
}

# Logging Settings
LOGGING = {
    "level": "INFO",
    "file": "voicebot.log",
    "console": True,
}
