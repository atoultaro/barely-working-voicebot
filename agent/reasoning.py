"""
Agent reasoning module for decision making and task planning.
Uses a language model to make decisions based on user intents and context.
"""
import logging
import json
from typing import Tuple, Dict, List, Any, Optional

import config
from nlu.understanding import NLUEngine
from agent.memory import AgentMemory

logger = logging.getLogger(__name__)

class Agent:
    """
    Agent for decision making and task planning.
    """
    
    def __init__(self, nlu_engine: NLUEngine):
        """
        Initialize the agent.
        
        Args:
            nlu_engine: NLU engine for language understanding and generation
        """
        self.config = config.AGENT
        self.nlu = nlu_engine
        self.memory = AgentMemory()
        logger.info("Agent initialized")
    
    def decide(self, intent: str, entities: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """
        Make a decision based on the user's intent and entities.
        
        Args:
            intent: The identified user intent
            entities: Dictionary of extracted entities
            
        Returns:
            Tuple containing:
                - response: The response to give to the user
                - action: Action to take (or None if no action needed)
                - emotion: Appropriate emotion for the response
        """
        logger.info(f"Agent deciding on intent: {intent}")
        
        # Update agent memory with new information
        self.memory.update_from_interaction(intent, entities)
        
        # Check confidence in understanding
        if intent == "unknown" and self._should_clarify():
            # Ask for clarification
            response, emotion = self._generate_clarification()
            return response, None, emotion
        
        # Handle different intents
        if intent == "greeting":
            return self._handle_greeting(entities)
            
        elif intent == "farewell":
            return self._handle_farewell(entities)
            
        elif intent == "information_request":
            return self._handle_information_request(entities)
            
        elif intent == "action_request":
            return self._handle_action_request(entities)
            
        elif intent == "task_specific":
            return self._handle_task_specific(entities)
            
        else:
            # For other intents, generate a contextual response
            response, emotion = self.nlu.generate_response(intent, entities)
            return response, None, emotion
    
    def handle_result(self, result: Dict[str, Any]) -> Tuple[str, str]:
        """
        Handle the result of an action execution.
        
        Args:
            result: Result from action execution
            
        Returns:
            Tuple containing:
                - response: The response to give to the user
                - emotion: Appropriate emotion for the response
        """
        # Update memory with action result
        self.memory.update_from_action_result(result)
        
        # Generate response based on the result
        success = result.get("success", False)
        
        if success:
            intent = "action_success"
            emotion = "happy"
        else:
            intent = "action_failure"
            emotion = "concerned"
        
        response, generated_emotion = self.nlu.generate_response(
            intent, 
            {}, 
            action_result=result
        )
        
        return response, generated_emotion or emotion
    
    def _should_clarify(self) -> bool:
        """
        Determine if clarification is needed.
        
        Returns:
            True if clarification is needed, False otherwise
        """
        # Check if we've already asked for clarification too many times
        clarification_count = self.memory.get_recent_clarification_count()
        return clarification_count < self.config["max_clarification_turns"]
    
    def _generate_clarification(self) -> Tuple[str, str]:
        """
        Generate a clarification request.
        
        Returns:
            Tuple containing:
                - response: Clarification request text
                - emotion: Appropriate emotion
        """
        self.memory.increment_clarification_count()
        
        # Generate contextual clarification request
        return self.nlu.generate_response(
            "request_clarification",
            {"reason": "unclear_intent"}
        )
    
    def _handle_greeting(self, entities: Dict[str, Any]) -> Tuple[str, None, str]:
        """
        Handle greeting intent.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Tuple of response, action, and emotion
        """
        # Check if this is a first-time greeting or a repeat
        if self.memory.is_first_interaction():
            response = "Hello! I'm your emotional voicebot assistant. How can I help you today?"
            emotion = "happy"
        else:
            # Generate contextual greeting
            response, emotion = self.nlu.generate_response("greeting", entities)
        
        return response, None, emotion
    
    def _handle_farewell(self, entities: Dict[str, Any]) -> Tuple[str, None, str]:
        """
        Handle farewell intent.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Tuple of response, action, and emotion
        """
        response, emotion = self.nlu.generate_response("farewell", entities)
        return response, None, emotion
    
    def _handle_information_request(self, entities: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """
        Handle information request intent.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Tuple of response, action, and emotion
        """
        # Determine if we need to fetch information via MCP
        topic = entities.get("topic")
        
        if topic and self._requires_external_information(topic):
            # Create action to fetch information
            action = {
                "type": "fetch_information",
                "parameters": {
                    "topic": topic,
                    "filters": entities.get("filters", {})
                }
            }
            
            # Generate waiting response
            response, emotion = self.nlu.generate_response(
                "information_request_processing",
                entities
            )
            
            return response, action, emotion
        else:
            # Generate response from existing knowledge
            response, emotion = self.nlu.generate_response(
                "information_provision",
                entities
            )
            
            return response, None, emotion
    
    def _handle_action_request(self, entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
        """
        Handle action request intent.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Tuple of response, action, and emotion
        """
        # Extract action details
        action_type = entities.get("action_type", "unknown")
        
        # Create action object
        action = {
            "type": action_type,
            "parameters": {k: v for k, v in entities.items() if k != "action_type"}
        }
        
        # Generate acknowledgment response
        response, emotion = self.nlu.generate_response(
            "action_acknowledgment",
            {"action_type": action_type}
        )
        
        return response, action, emotion
    
    def _handle_task_specific(self, entities: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """
        Handle task-specific intent.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Tuple of response, action, and emotion
        """
        task = entities.get("task")
        
        if not task:
            # Not enough information to determine the task
            response, emotion = self._generate_clarification()
            return response, None, emotion
        
        # Check if task requires an action
        if self._requires_action(task, entities):
            # Create task-specific action
            action = {
                "type": "execute_task",
                "parameters": {
                    "task": task,
                    "details": entities
                }
            }
            
            # Generate acknowledgment
            response, emotion = self.nlu.generate_response(
                "task_acknowledgment",
                {"task": task}
            )
            
            return response, action, emotion
        else:
            # Generate informational response about the task
            response, emotion = self.nlu.generate_response(
                "task_information",
                {"task": task}
            )
            
            return response, None, emotion
    
    def _requires_external_information(self, topic: str) -> bool:
        """
        Determine if a topic requires fetching external information.
        
        Args:
            topic: The information topic
            
        Returns:
            True if external information is needed, False otherwise
        """
        # Check if we already have information on this topic in memory
        return not self.memory.has_information_on_topic(topic)
    
    def _requires_action(self, task: str, details: Dict[str, Any]) -> bool:
        """
        Determine if a task requires an action to be executed.
        
        Args:
            task: The task name
            details: Task details
            
        Returns:
            True if an action is required, False otherwise
        """
        # Simple heuristic: tasks with verbs like "send", "create", "update" require actions
        action_verbs = ["send", "create", "update", "delete", "modify", "execute", "run", "start", "stop"]
        
        # Check if task contains an action verb
        return any(verb in task.lower() for verb in action_verbs)
