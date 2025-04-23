# Emotional Voicebot with Agent Capabilities

A sophisticated voicebot system that:
1. Understands spoken English through speech recognition
2. Makes intelligent decisions like an agent
3. Executes tasks through Model Context Protocol (MCP)
4. Responds with emotionally appropriate synthesized speech

## Features

- **Speech Recognition**: Real-time conversion of spoken English to text
- **Natural Language Understanding**: Interpretation of user intent and context
- **Agent Decision Making**: Intelligent reasoning and action planning
- **MCP Integration**: Task execution through Model Context Protocol
- **Emotional Speech Synthesis**: Natural-sounding responses with appropriate emotions

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
# Add other API keys as needed
```

3. Run the voicebot:
```
python main.py
```

## Project Structure

- `speech/`: Speech recognition and synthesis modules
- `nlu/`: Natural language understanding components
- `agent/`: Decision-making and reasoning system
- `mcp/`: Model Context Protocol integration
- `utils/`: Utility functions and helpers

## Configuration

Adjust settings in `config.py` to customize:
- Voice characteristics
- Emotional response parameters
- Agent behavior
- MCP endpoints
