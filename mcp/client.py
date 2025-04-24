"""
Model Context Protocol (MCP) client for task execution.
Implements a client for communicating with external systems through MCP.
"""
import logging
import json
import time
import asyncio
import threading
from typing import Dict, Any, Optional, Callable

import websockets
import requests

import config
from mcp.handlers import get_handler_for_action

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for executing tasks through the Model Context Protocol.
    """
    
    def __init__(self):
        """Initialize the MCP client with configured settings."""
        self.config = config.MCP
        self.ws_client = None
        self.ws_connected = False
        self.ws_thread = None
        self.message_queue = asyncio.Queue()
        self.response_futures = {}
        self.next_message_id = 1
        self.connection_failed = False  # Track if connection has failed before
        self.last_connection_attempt = 0  # Track when we last attempted to connect
        self.recent_failure_timeout = 60  # Timeout for recent connection failures
        logger.info("MCP client initialized")
    
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action through MCP.
        
        Args:
            action: The action to execute
            
        Returns:
            Result of the action execution
        """
        logger.info(f"Executing action: {action['type']}")
        
        # Get the appropriate handler for this action
        handler = get_handler_for_action(action["type"])
        
        if handler:
            # Use local handler if available
            try:
                result = handler.handle(action["parameters"])
                logger.info(f"Action {action['type']} executed locally")
                return result
            except Exception as e:
                logger.error(f"Error executing action locally: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        else:
            # Use MCP to execute the action
            return self._execute_via_mcp(action)
    
    def _execute_via_mcp(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action via MCP.
        
        Args:
            action: The action to execute
            
        Returns:
            Result of the action execution
        """
        # Check if we're using WebSocket or HTTP
        if self.config["endpoint"].startswith("ws"):
            return self._execute_via_websocket(action)
        else:
            return self._execute_via_http(action)
    
    def _execute_via_http(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action via HTTP.
        
        Args:
            action: The action to execute
            
        Returns:
            Result of the action execution
        """
        try:
            # Prepare request
            endpoint = self.config["endpoint"]
            headers = {"Content-Type": "application/json"}
            payload = {
                "action": action["type"],
                "parameters": action["parameters"]
            }
            
            # Send request with retry logic
            for attempt in range(self.config["retry_attempts"]):
                try:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.config["timeout"]
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.warning(
                            f"HTTP request failed with status {response.status_code}: {response.text}"
                        )
                        
                except requests.RequestException as e:
                    logger.warning(f"HTTP request attempt {attempt+1} failed: {e}")
                
                # Wait before retrying
                if attempt < self.config["retry_attempts"] - 1:
                    time.sleep(self.config["retry_delay"])
            
            # All attempts failed
            return {
                "success": False,
                "error": "Failed to execute action after multiple attempts"
            }
            
        except Exception as e:
            logger.error(f"Error executing action via HTTP: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_via_websocket(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action via WebSocket.
        
        Args:
            action: The action to execute
            
        Returns:
            Result of the action execution
        """
        # Check if we've had connection failures recently
        current_time = time.time()
        if self.connection_failed and (current_time - self.last_connection_attempt < self.recent_failure_timeout):  
            logger.info("Skipping WebSocket connection attempt due to recent failure")
            return {
                "success": False,
                "error": "WebSocket connection unavailable (skipping retry)"
            }
            
        # Ensure WebSocket connection is established
        if not self.ws_connected:
            self.last_connection_attempt = current_time
            self._connect_websocket()
            
            # Wait for connection to establish
            start_time = time.time()
            while not self.ws_connected and time.time() - start_time < 5:
                time.sleep(0.1)
                
            if not self.ws_connected:
                self.connection_failed = True  # Mark that connection has failed
                return {
                    "success": False,
                    "error": "Failed to establish WebSocket connection"
                }
        
        # Connection succeeded, reset failure flag
        self.connection_failed = False
        
        # Create message
        message_id = self._get_next_message_id()
        message = {
            "id": message_id,
            "action": action["type"],
            "parameters": action["parameters"]
        }
        
        # Create future for response
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        self.response_futures[message_id] = future
        
        # Send message
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put(json.dumps(message)),
            self.ws_thread.loop
        )
        
        try:
            # Wait for response with timeout
            result = asyncio.run(self._wait_for_response(future))
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response to message {message_id}")
            return {
                "success": False,
                "error": "Timeout waiting for response"
            }
        finally:
            # Clean up
            if message_id in self.response_futures:
                del self.response_futures[message_id]
    
    async def _wait_for_response(self, future):
        """
        Wait for a response to a WebSocket message.
        
        Args:
            future: Future to wait for
            
        Returns:
            Response data
        """
        return await asyncio.wait_for(future, timeout=self.config["timeout"])
    
    def _connect_websocket(self):
        """Establish WebSocket connection in a separate thread."""
        if self.ws_thread and self.ws_thread.is_alive():
            return
        
        self.ws_thread = WebSocketThread(
            self.config["endpoint"],
            self.message_queue,
            self._handle_ws_message,
            self._handle_ws_connection_change
        )
        self.ws_thread.start()
    
    def _handle_ws_message(self, message: str):
        """
        Handle incoming WebSocket message.
        
        Args:
            message: The received message
        """
        try:
            data = json.loads(message)
            message_id = data.get("id")
            
            if message_id and message_id in self.response_futures:
                future = self.response_futures[message_id]
                asyncio.run_coroutine_threadsafe(
                    self._set_future_result(future, data),
                    self.ws_thread.loop
                )
                
        except json.JSONDecodeError:
            logger.error(f"Received invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _set_future_result(self, future, result):
        """
        Set the result of a future.
        
        Args:
            future: The future to set the result for
            result: The result to set
        """
        if not future.done():
            future.set_result(result)
    
    def _handle_ws_connection_change(self, connected: bool):
        """
        Handle WebSocket connection state change.
        
        Args:
            connected: Whether the connection is established
        """
        self.ws_connected = connected
        if connected:
            logger.info("WebSocket connection established")
        else:
            logger.info("WebSocket connection closed")
    
    def _get_next_message_id(self) -> int:
        """
        Get the next message ID.
        
        Returns:
            Next message ID
        """
        message_id = self.next_message_id
        self.next_message_id += 1
        return message_id
    
    def close(self):
        """Close the MCP client and clean up resources."""
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.stop()
            self.ws_thread.join(timeout=2)
        
        self.ws_connected = False
        logger.info("MCP client closed")


class WebSocketThread(threading.Thread):
    """Thread for handling WebSocket communication."""
    
    def __init__(self, endpoint: str, message_queue: asyncio.Queue,
                message_handler: Callable[[str], None],
                connection_handler: Callable[[bool], None]):
        """
        Initialize the WebSocket thread.
        
        Args:
            endpoint: WebSocket endpoint URL
            message_queue: Queue for outgoing messages
            message_handler: Handler for incoming messages
            connection_handler: Handler for connection state changes
        """
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self.message_queue = message_queue
        self.message_handler = message_handler
        self.connection_handler = connection_handler
        self.running = True
        self.loop = None
        self.ws = None
    
    def run(self):
        """Run the WebSocket thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._run_websocket())
        finally:
            self.loop.close()
    
    async def _run_websocket(self):
        """Run the WebSocket client."""
        retry_delay = 1
        max_retry_delay = 30
        
        while self.running:
            try:
                async with websockets.connect(self.endpoint) as ws:
                    self.ws = ws
                    self.connection_handler(True)
                    retry_delay = 1  # Reset retry delay on successful connection
                    
                    # Start send and receive tasks
                    receive_task = asyncio.create_task(self._receive_messages(ws))
                    send_task = asyncio.create_task(self._send_messages(ws))
                    
                    # Wait for either task to complete
                    done, pending = await asyncio.wait(
                        [receive_task, send_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                    
                    # Handle results
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            logger.error(f"WebSocket task error: {e}")
                    
                    self.ws = None
                    self.connection_handler(False)
                    
            except (websockets.exceptions.WebSocketException, ConnectionError) as e:
                logger.warning(f"WebSocket connection error: {e}")
                self.connection_handler(False)
                
            except Exception as e:
                logger.error(f"Unexpected WebSocket error: {e}")
                self.connection_handler(False)
            
            # Only retry if still running
            if self.running:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
    
    async def _receive_messages(self, ws):
        """
        Receive messages from the WebSocket.
        
        Args:
            ws: WebSocket connection
        """
        async for message in ws:
            if not self.running:
                break
                
            try:
                self.message_handler(message)
            except Exception as e:
                logger.error(f"Error handling received message: {e}")
    
    async def _send_messages(self, ws):
        """
        Send messages to the WebSocket.
        
        Args:
            ws: WebSocket connection
        """
        while self.running:
            try:
                message = await self.message_queue.get()
                await ws.send(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                break
    
    def stop(self):
        """Stop the WebSocket thread."""
        self.running = False
        if self.ws:
            asyncio.run_coroutine_threadsafe(
                self.ws.close(),
                self.loop
            )
