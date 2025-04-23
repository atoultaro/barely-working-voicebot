"""
Conversation context management for the NLU system.
Maintains a history of conversation turns for contextual understanding.
"""
import logging
from typing import List, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

class ConversationContext:
    """
    Manages conversation history for contextual understanding.
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize the conversation context.
        
        Args:
            max_turns: Maximum number of conversation turns to remember
        """
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns * 2)  # Each turn has user + assistant message
        logger.info(f"Conversation context initialized with {max_turns} turns capacity")
    
    def add_user_message(self, message: str):
        """
        Add a user message to the conversation history.
        
        Args:
            message: The user's message
        """
        if message:
            self.history.append({"role": "user", "content": message})
            logger.debug(f"Added user message to context: {message[:50]}...")
    
    def add_assistant_message(self, message: str):
        """
        Add an assistant message to the conversation history.
        
        Args:
            message: The assistant's message
        """
        if message:
            self.history.append({"role": "assistant", "content": message})
            logger.debug(f"Added assistant message to context: {message[:50]}...")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return list(self.history)
    
    def get_formatted_history(self) -> str:
        """
        Get the conversation history formatted as a string.
        
        Returns:
            Formatted conversation history
        """
        formatted = []
        for message in self.history:
            role = message["role"].capitalize()
            content = message["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Clear the conversation history."""
        self.history.clear()
        logger.info("Conversation context cleared")
    
    def get_last_user_message(self) -> str:
        """
        Get the last message from the user.
        
        Returns:
            The last user message or empty string if none exists
        """
        for message in reversed(self.history):
            if message["role"] == "user":
                return message["content"]
        return ""
    
    def get_last_assistant_message(self) -> str:
        """
        Get the last message from the assistant.
        
        Returns:
            The last assistant message or empty string if none exists
        """
        for message in reversed(self.history):
            if message["role"] == "assistant":
                return message["content"]
        return ""
