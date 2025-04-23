"""
Agent memory module for storing and retrieving information from interactions.
Maintains a memory of past interactions, learned facts, and user preferences.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AgentMemory:
    """
    Manages the agent's memory of interactions, facts, and preferences.
    """
    
    def __init__(self):
        """Initialize the agent memory."""
        # Recent interactions
        self.recent_interactions = deque(maxlen=20)
        
        # Learned facts by topic
        self.facts = defaultdict(dict)
        
        # User preferences
        self.user_preferences = {}
        
        # Clarification tracking
        self.clarification_count = 0
        self.last_clarification_time = 0
        
        # Interaction counter
        self.interaction_count = 0
        
        logger.info("Agent memory initialized")
    
    def update_from_interaction(self, intent: str, entities: Dict[str, Any]):
        """
        Update memory based on a new interaction.
        
        Args:
            intent: The identified user intent
            entities: Dictionary of extracted entities
        """
        # Record the interaction
        interaction = {
            "timestamp": time.time(),
            "intent": intent,
            "entities": entities
        }
        self.recent_interactions.append(interaction)
        
        # Extract and store facts from entities
        self._extract_facts(entities)
        
        # Extract and store preferences
        self._extract_preferences(intent, entities)
        
        # Update interaction counter
        self.interaction_count += 1
        
        logger.debug(f"Memory updated from interaction with intent: {intent}")
    
    def update_from_action_result(self, result: Dict[str, Any]):
        """
        Update memory based on an action result.
        
        Args:
            result: Result from action execution
        """
        # Record the action result
        action_result = {
            "timestamp": time.time(),
            "result": result
        }
        self.recent_interactions.append(action_result)
        
        # Extract and store facts from result
        if "data" in result:
            self._extract_facts_from_result(result["data"])
        
        logger.debug("Memory updated from action result")
    
    def get_recent_clarification_count(self) -> int:
        """
        Get the number of recent clarification requests.
        
        Returns:
            Number of clarification requests in the recent window
        """
        # Reset clarification count if it's been a while
        if time.time() - self.last_clarification_time > 300:  # 5 minutes
            self.clarification_count = 0
        
        return self.clarification_count
    
    def increment_clarification_count(self):
        """Increment the clarification counter."""
        self.clarification_count += 1
        self.last_clarification_time = time.time()
    
    def is_first_interaction(self) -> bool:
        """
        Check if this is the first interaction with the user.
        
        Returns:
            True if this is the first interaction, False otherwise
        """
        return self.interaction_count == 0
    
    def has_information_on_topic(self, topic: str) -> bool:
        """
        Check if we have information on a specific topic.
        
        Args:
            topic: The topic to check
            
        Returns:
            True if we have information, False otherwise
        """
        return topic.lower() in self.facts
    
    def get_facts_about_topic(self, topic: str) -> Dict[str, Any]:
        """
        Get all facts about a specific topic.
        
        Args:
            topic: The topic to get facts about
            
        Returns:
            Dictionary of facts about the topic
        """
        return self.facts.get(topic.lower(), {})
    
    def get_user_preference(self, preference_name: str) -> Optional[Any]:
        """
        Get a user preference value.
        
        Args:
            preference_name: Name of the preference
            
        Returns:
            Preference value or None if not set
        """
        return self.user_preferences.get(preference_name)
    
    def get_recent_interactions(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent interactions.
        
        Args:
            count: Number of recent interactions to retrieve
            
        Returns:
            List of recent interactions
        """
        return list(self.recent_interactions)[-count:]
    
    def _extract_facts(self, entities: Dict[str, Any]):
        """
        Extract facts from entities.
        
        Args:
            entities: Dictionary of extracted entities
        """
        # Look for topic-related entities
        topic = entities.get("topic")
        if topic:
            topic = topic.lower()
            
            # Store all other entities as facts about this topic
            for key, value in entities.items():
                if key != "topic":
                    self.facts[topic][key] = value
                    logger.debug(f"Stored fact: {topic}.{key} = {value}")
    
    def _extract_facts_from_result(self, data: Dict[str, Any]):
        """
        Extract facts from action result data.
        
        Args:
            data: Data from action result
        """
        # Look for topic-related data
        topic = data.get("topic")
        if topic:
            topic = topic.lower()
            
            # Store all other data as facts about this topic
            for key, value in data.items():
                if key != "topic":
                    self.facts[topic][key] = value
                    logger.debug(f"Stored fact from result: {topic}.{key} = {value}")
    
    def _extract_preferences(self, intent: str, entities: Dict[str, Any]):
        """
        Extract user preferences from intent and entities.
        
        Args:
            intent: The identified user intent
            entities: Dictionary of extracted entities
        """
        # Check for preference-setting intent
        if intent == "set_preference":
            preference_name = entities.get("preference_name")
            preference_value = entities.get("preference_value")
            
            if preference_name and preference_value is not None:
                self.user_preferences[preference_name] = preference_value
                logger.info(f"Stored user preference: {preference_name} = {preference_value}")
        
        # Look for implicit preferences
        elif "preference" in entities:
            preferences = entities["preference"]
            if isinstance(preferences, dict):
                for name, value in preferences.items():
                    self.user_preferences[name] = value
                    logger.info(f"Stored implicit user preference: {name} = {value}")
    
    def clear(self):
        """Clear all memory."""
        self.recent_interactions.clear()
        self.facts.clear()
        self.user_preferences.clear()
        self.clarification_count = 0
        self.last_clarification_time = 0
        self.interaction_count = 0
        logger.info("Agent memory cleared")
