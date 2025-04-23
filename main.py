#!/usr/bin/env python3
"""
Main entry point for the Emotional Voicebot with Agent Capabilities.
"""
import os
import time
import logging
import signal
import sys
from dotenv import load_dotenv

# Set up logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('voicebot.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import voicebot components
try:
    from speech.recognition import SpeechRecognizer
    from speech.synthesis import SpeechSynthesizer
    from nlu.understanding import NLUEngine
    from agent.reasoning import Agent
    from mcp.client import MCPClient
    from utils.audio import AudioManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Make sure you've installed all dependencies with: pip install -r requirements.txt")
    sys.exit(1)

class Voicebot:
    """Main voicebot class that coordinates all components."""
    
    def __init__(self):
        """Initialize the voicebot and all its components."""
        logger.info("Initializing voicebot...")
        
        # Initialize components
        self.audio_manager = AudioManager()
        self.speech_recognizer = SpeechRecognizer()
        self.nlu_engine = NLUEngine()
        self.agent = Agent(self.nlu_engine)
        self.mcp_client = MCPClient()
        self.speech_synthesizer = SpeechSynthesizer()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        self.running = False
        logger.info("Voicebot initialized successfully")
    
    def start(self):
        """Start the voicebot and begin listening for commands."""
        logger.info("Starting voicebot...")
        self.running = True
        
        # Greeting
        greeting = "Hello! I'm your emotional voicebot assistant. How can I help you today?"
        self._speak(greeting, emotion="greeting")
        
        try:
            while self.running:
                # Listen for speech
                logger.info("Listening for speech...")
                speech_text = self.speech_recognizer.listen()
                
                if not speech_text:
                    continue
                
                logger.info(f"Recognized: {speech_text}")
                
                # Process through NLU
                intent, entities = self.nlu_engine.process(speech_text)
                logger.info(f"Intent: {intent}, Entities: {entities}")
                
                # Let agent make decisions
                response, action, emotion = self.agent.decide(intent, entities)
                
                # Execute actions through MCP if needed
                if action:
                    logger.info(f"Executing action: {action}")
                    result = self.mcp_client.execute(action)
                    
                    # Update response based on action result
                    response, emotion = self.agent.handle_result(result)
                
                # Respond with emotional speech
                if response:
                    self._speak(response, emotion)
                
                # Small pause between interactions
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in voicebot main loop: {e}")
            self.stop()
    
    def _speak(self, text, emotion=None):
        """Generate and play emotional speech."""
        logger.info(f"Speaking with emotion '{emotion}': {text}")
        self.speech_synthesizer.speak(text, emotion=emotion)
    
    def stop(self):
        """Stop the voicebot and clean up resources."""
        logger.info("Stopping voicebot...")
        self.running = False
        
        # Clean up resources
        self.speech_recognizer.close()
        self.speech_synthesizer.close()
        self.mcp_client.close()
        self.audio_manager.close()
        
        logger.info("Voicebot stopped")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

if __name__ == "__main__":
    # Check for required API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("ELEVENLABS_API_KEY"):
        missing_keys.append("ELEVENLABS_API_KEY")
    
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        logger.info("Please add them to your .env file")
        sys.exit(1)
    
    # Create and start the voicebot
    bot = Voicebot()
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        bot.stop()
