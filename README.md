# Multilingual AI Voice Assistant

An AI-powered voice assistant that can understand and respond in multiple languages while maintaining a specific character personality.

## Features
- Voice input and output
- Multilingual support
- Character-based responses
- Real-time language detection and translation
- Modern web interface

## Prerequisites
1. Python 3.8 or higher
2. Ollama installed and running locally with a Llama model
   - Install Ollama from: https://ollama.ai/
   - Pull the Llama model: `ollama pull llama2`

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure Ollama is running locally with your Llama model

3. Run the application:
   ```bash
   python app.py
   ```

## Character Profiles

The AI assistant can take on different character personalities:

1. **Rude Banker**: A sarcastic and impatient banker who reluctantly helps customers
2. **Humble Actor**: A friendly and enthusiastic actor who loves interacting with fans
3. **Sassy Chef**: A passionate chef who's always ready to share cooking tips

## Technologies Used
- Flask (Web Framework)
- SpeechRecognition (Voice Input)
- pyttsx3 (Text-to-Speech)
- langdetect (Language Detection)
- googletrans (Translation)
- Ollama (Local LLM) 