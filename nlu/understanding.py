"""
Natural Language Understanding module for interpreting user speech.
Uses OpenAI's language models to extract intents and entities.
"""
import logging
import json
from typing import Tuple, Dict, List, Any, Optional

import openai
from openai import OpenAI

import config
from nlu.context import ConversationContext

logger = logging.getLogger(__name__)

class NLUEngine:
    """
    Handles natural language understanding using OpenAI's language models.
    Extracts intents and entities from user speech and maintains conversation context.
    """
    
    def __init__(self):
        """Initialize the NLU engine with configured settings."""
        self.config = config.NLU
        self.client = OpenAI()
        self.context = ConversationContext(max_turns=self.config["context_window"])
        logger.info("NLU engine initialized")
    
    def process(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process text to extract intent and entities.
        
        Args:
            text: User's speech text
            
        Returns:
            Tuple containing:
                - intent: The identified user intent
                - entities: Dictionary of extracted entities
        """
        if not text:
            return "null", {}
        
        # Add user message to context
        self.context.add_user_message(text)
        
        try:
            # Create prompt for intent and entity extraction
            system_prompt = """
            You are an AI assistant that analyzes user speech to extract intent and entities.
            Output ONLY a JSON object with the following structure:
            {
                "intent": "the_primary_intent",
                "entities": {
                    "entity_name": "entity_value",
                    ...
                }
            }
            
            Common intents include:
            - greeting: User is greeting the system
            - information_request: User is asking for information
            - action_request: User wants the system to perform an action
            - clarification: User is asking for clarification
            - confirmation: User is confirming something
            - rejection: User is rejecting something
            - farewell: User is saying goodbye
            - smalltalk: User is making small talk
            - task_specific: User is referring to a specific task (specify in entities)
            - personal_question: User is asking about you personally
            - opinion_request: User is asking for your opinion
            - feedback: User is providing feedback
            
            Be flexible in your intent classification. Don't overclassify casual questions as philosophical inquiries about AI personhood.
            Focus on the practical intent behind the user's words rather than literal interpretation.
            
            Extract all relevant entities from the user's speech.
            """
            
            # Get conversation history for context
            conversation_history = self.context.get_formatted_history()
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.config["model"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation history:\n{conversation_history}\n\nCurrent user message: {text}"}
                ]
            )
            
            # Extract and parse response
            content = response.choices[0].message.content.strip()
            
            # Handle potential non-JSON responses
            try:
                result = json.loads(content)
                intent = result.get("intent", "unknown")
                entities = result.get("entities", {})
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse NLU response as JSON: {content}")
                intent = "unknown"
                entities = {}
            
            logger.info(f"Extracted intent: {intent}, entities: {entities}")
            return intent, entities
            
        except Exception as e:
            logger.error(f"Error in NLU processing: {e}")
            return "error", {"error_message": str(e)}
    
    def generate_response(self, intent: str, entities: Dict[str, Any], 
                         action_result: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Generate a natural language response based on intent, entities, and action results.
        
        Args:
            intent: The identified user intent
            entities: Dictionary of extracted entities
            action_result: Optional result from an action execution
            
        Returns:
            Tuple containing:
                - response: The generated response text
                - emotion: The appropriate emotion for the response
        """
        try:
            # Create prompt for response generation
            system_prompt = """
            You are an emotional AI assistant that generates natural, conversational responses.
            Your responses should be concise, helpful, and emotionally appropriate.
            
            IMPORTANT GUIDELINES:
            1. Avoid philosophical discussions about AI consciousness or personhood unless explicitly asked
            2. Focus on being helpful and addressing the user's actual needs
            3. Vary your responses - don't repeat the same phrases
            4. Be conversational and natural, not robotic
            5. Keep responses brief but informative
            6. Don't start responses with phrases like "As an AI assistant..."
            
            Output ONLY a JSON object with the following structure:
            {
                "response": "Your natural language response here",
                "emotion": "appropriate_emotion"
            }
            
            Available emotions: neutral, happy, sad, angry, surprised, concerned
            
            Choose the most appropriate emotion based on the context and content of your response.
            """
            
            # Prepare input for the model
            input_content = {
                "intent": intent,
                "entities": entities,
                "conversation_history": self.context.get_formatted_history(),
                "action_result": action_result
            }
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.config["model"],
                temperature=0.8,  # Slightly increased for more variation
                max_tokens=self.config["max_tokens"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a response for: {json.dumps(input_content)}"}
                ]
            )
            
            # Extract and parse response
            content = response.choices[0].message.content.strip()
            
            # Handle potential non-JSON responses
            try:
                result = json.loads(content)
                response_text = result.get("response", "I'm not sure how to respond to that.")
                emotion = result.get("emotion", "neutral")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse response generation as JSON: {content}")
                response_text = content if content else "I'm not sure how to respond to that."
                emotion = "neutral"
            
            # Add assistant message to context
            self.context.add_assistant_message(response_text)
            
            logger.info(f"Generated response with emotion '{emotion}': {response_text}")
            return response_text, emotion
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return "I'm having trouble processing that right now.", "concerned"
