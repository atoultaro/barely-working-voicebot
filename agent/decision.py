"""
Agent decision module for making decisions about actions to take.
Implements decision-making logic based on user intents and available actions.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Tuple

import config

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Makes decisions about what actions to take based on user intents and context.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        self.config = config.AGENT
        self.available_actions = self._load_available_actions()
        logger.info("Decision engine initialized")
    
    def decide_action(self, intent: str, entities: Dict[str, Any], 
                     context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decide what action to take based on intent, entities, and context.
        
        Args:
            intent: The identified user intent
            entities: Dictionary of extracted entities
            context: Additional context information
            
        Returns:
            Action to take or None if no action is needed
        """
        logger.info(f"Deciding action for intent: {intent}")
        
        # Handle language change intent specially - no need to use MCP
        if intent == "language_change":
            logger.info(f"Language change intent detected: {entities.get('language', 'unknown')}")
            # Return None to indicate no external action needed
            # The language handling is done at the NLU and TTS/STT level
            return None
        
        # Check if intent directly maps to an action
        if intent in self.available_actions:
            return self._create_action_from_intent(intent, entities)
        
        # Handle special intents
        if intent == "action_request":
            return self._handle_action_request(entities)
        
        if intent == "information_request":
            return self._handle_information_request(entities, context)
        
        if intent == "task_specific":
            return self._handle_task_specific(entities, context)
        
        # No action needed for other intents
        return None
    
    def validate_action(self, action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate that an action is properly formed and can be executed.
        
        Args:
            action: The action to validate
            
        Returns:
            Tuple containing:
                - valid: True if the action is valid, False otherwise
                - error: Error message if invalid, None otherwise
        """
        # Check for required fields
        if "type" not in action:
            return False, "Action is missing required 'type' field"
        
        action_type = action["type"]
        
        # Check if action type is supported
        if action_type not in self.available_actions:
            return False, f"Unsupported action type: {action_type}"
        
        # Check for required parameters
        required_params = self.available_actions[action_type].get("required_params", [])
        parameters = action.get("parameters", {})
        
        for param in required_params:
            if param not in parameters:
                return False, f"Action is missing required parameter: {param}"
        
        return True, None
    
    def _load_available_actions(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the available actions and their metadata.
        
        Returns:
            Dictionary of available actions
        """
        # In a real implementation, this might load from a configuration file
        return {
            "fetch_information": {
                "description": "Fetch information on a topic",
                "required_params": ["topic"],
                "optional_params": ["filters"]
            },
            "send_message": {
                "description": "Send a message",
                "required_params": ["recipient", "content"],
                "optional_params": ["priority"]
            },
            "create_item": {
                "description": "Create a new item",
                "required_params": ["item_type", "details"],
                "optional_params": []
            },
            "update_item": {
                "description": "Update an existing item",
                "required_params": ["item_id", "updates"],
                "optional_params": []
            },
            "delete_item": {
                "description": "Delete an item",
                "required_params": ["item_id"],
                "optional_params": ["confirm"]
            },
            "execute_task": {
                "description": "Execute a specific task",
                "required_params": ["task"],
                "optional_params": ["details"]
            },
            "search": {
                "description": "Search for information",
                "required_params": ["query"],
                "optional_params": ["filters", "limit"]
            }
        }
    
    def _create_action_from_intent(self, intent: str, 
                                  entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an action based on a direct intent-to-action mapping.
        
        Args:
            intent: The intent that maps to an action
            entities: Dictionary of extracted entities
            
        Returns:
            Action to take
        """
        # Create basic action structure
        action = {
            "type": intent,
            "parameters": {}
        }
        
        # Add parameters from entities
        action_metadata = self.available_actions[intent]
        required_params = action_metadata.get("required_params", [])
        optional_params = action_metadata.get("optional_params", [])
        
        for param in required_params + optional_params:
            if param in entities:
                action["parameters"][param] = entities[param]
        
        return action
    
    def _handle_action_request(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an explicit action request.
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Action to take or None if no action is needed
        """
        # Handle content generation tasks locally (poems, stories, etc.)
        task = entities.get("task")
        task_specific = entities.get("task_specific", "")
        
        # Check if this is a content generation task that can be handled locally
        content_generation_tasks = ["recite a poem", "tell a story", "sing a song", "read poetry"]
        
        # Check for exact matches in task
        if task and any(t == task for t in content_generation_tasks):
            logger.info(f"Content generation request detected: {task}")
            # Return None to indicate no external action needed
            return None
            
        # Check for keywords in task_specific
        poem_keywords = ["poem", "poetry", "诗", "朗读", "念", "recite", "read"]
        story_keywords = ["story", "tale", "narrative", "故事"]
        song_keywords = ["song", "sing", "歌"]
        summary_keywords = ["summary", "summarize", "summarization", "overview", "brief"]
        
        if task_specific and any(keyword in task_specific.lower() for keyword in poem_keywords + story_keywords + song_keywords + summary_keywords):
            logger.info(f"Content generation request detected via keywords: {task_specific}")
            # Return None to indicate no external action needed
            return None
            
        # Also check if task is simply "summary" or similar
        if task and any(keyword in task.lower() for keyword in summary_keywords):
            logger.info(f"Summary request detected: {task}")
            # Return None to indicate no external action needed
            return None
        
        action_type = entities.get("action_type")
        
        if not action_type or action_type not in self.available_actions:
            # Default to a generic action if not specified
            action_type = "execute_task"
        
        # Create action with all entities as parameters
        action = {
            "type": action_type,
            "parameters": {k: v for k, v in entities.items() if k != "action_type"}
        }
        
        return action
    
    def _handle_information_request(self, entities: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle an information request.
        
        Args:
            entities: Dictionary of extracted entities
            context: Additional context information
            
        Returns:
            Action to take or None if no action is needed
        """
        topic = entities.get("topic")
        
        if not topic:
            return None
        
        # Check if we already have information on this topic in context
        known_topics = context.get("known_topics", [])
        if topic.lower() in known_topics:
            return None
        
        # Create action to fetch information
        return {
            "type": "fetch_information",
            "parameters": {
                "topic": topic,
                "filters": entities.get("filters", {})
            }
        }
    
    def _handle_task_specific(self, entities: Dict[str, Any], 
                             context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle a task-specific intent.
        
        Args:
            entities: Dictionary of extracted entities
            context: Additional context information
            
        Returns:
            Action to take or None if no action is needed
        """
        task = entities.get("task")
        
        if not task:
            return None
        
        # Map common tasks to action types
        task_to_action = {
            "send": "send_message",
            "create": "create_item",
            "update": "update_item",
            "delete": "delete_item",
            "search": "search",
            "find": "search"
        }
        
        # Find the appropriate action type based on the task
        action_type = None
        for key, value in task_to_action.items():
            if key in task.lower():
                action_type = value
                break
        
        if not action_type:
            # Default to generic task execution
            action_type = "execute_task"
        
        # Create action
        return {
            "type": action_type,
            "parameters": {
                "task": task,
                "details": entities
            }
        }
