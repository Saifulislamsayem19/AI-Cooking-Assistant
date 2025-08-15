# AI Cooking Assistant 🍳

The AI Cooking Assistant is a revolutionary culinary companion that bridges the gap between traditional cooking and modern AI technology. Simply snap a photo of your ingredients, and watch as computer vision identifies what you have available. Our advanced AI then crafts personalized recipes tailored to your preferences, complete with hands-free voice navigation for a seamless cooking experience.

<img width="1556" height="883" alt="image" src="https://github.com/user-attachments/assets/053923e3-e6ae-483b-8016-f7410f3f44a9" />

## ✨ Features

### 🔍 Smart Ingredient Recognition
- **Computer Vision**: Upload photos of your fridge, pantry, or ingredients
- **High Accuracy**: Uses GPT-4 Vision with enhanced prompts for precise food identification
- **Comprehensive Detection**: Identifies packaged goods, fresh produce, frozen items, and more

### 🍽️ AI Recipe Generation
- **Personalized Recipes**: Generate 3 unique recipes based on your available ingredients
- **Dietary Preferences**: Support for various dietary restrictions and preferences
- **Cuisine Flexibility**: Specify cuisine types or let the AI suggest diverse options

### 🎙️ Voice-Guided Cooking
- **Real-time Voice Commands**: Navigate recipes hands-free while cooking
- **Smart Recognition**: Natural language processing for cooking-specific commands
- **Timer Integration**: Voice-activated timers with countdown notifications

### 🔊 Text-to-Speech
- **Audio Instructions**: Convert recipe steps to natural-sounding speech
- **Multiple Voices**: Choose from various voice options
- **Cached Audio**: Optimized performance with intelligent caching

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **OpenAI GPT-4**: Advanced language model for recipe generation and vision
- **LangChain**: Framework for LLM application development with LCEL syntax
- **Whisper API**: Speech-to-text for voice command recognition
- **WebSockets**: Real-time communication for cooking sessions

### AI & ML
- **GPT-4 Vision**: For ingredient identification from images
- **GPT-4o-mini**: Optimized model for recipe generation
- **Whisper**: Automatic speech recognition
- **OpenAI TTS**: Text-to-speech synthesis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Modern web browser with microphone access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-cooking-assistant.git
   cd ai-cooking-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Create static directory**
   ```bash
   mkdir static
   # Add your frontend files here
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
ai-cooking-assistant/
├── main.py                 # Main FastAPI application
├── static/                 # Frontend files (HTML, CSS, JS)
├── .env                    # Environment variables
├── .env.example           # Environment template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/identify-ingredients` | Upload image to identify ingredients |
| `POST` | `/api/generate-recipes` | Generate recipes from ingredients |
| `POST` | `/api/start-cooking/{recipe_id}` | Start a cooking session |
| `POST` | `/api/tts` | Generate text-to-speech audio |
| `GET` | `/api/recipe/{recipe_id}` | Get recipe details |

### Real-time Features

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WebSocket` | `/ws/{session_id}` | Real-time cooking guidance |
| `POST` | `/api/voice-command/{session_id}` | Process voice commands |

## 🎯 Usage Examples

### 1. Ingredient Identification
```python
import requests

# Upload an image of your fridge/pantry
with open('fridge_photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/identify-ingredients',
        files={'file': f}
    )
    ingredients = response.json()['ingredients']
```

### 2. Recipe Generation
```python
recipe_request = {
    "ingredients": ["chicken breast", "broccoli", "rice"],
    "dietary_preferences": "low-carb",
    "cuisine_type": "Asian"
}

response = requests.post(
    'http://localhost:8000/api/generate-recipes',
    json=recipe_request
)
recipes = response.json()['recipes']
```

### 3. Voice Commands
Supported voice commands during cooking:
- "Next step" - Move to next instruction
- "Previous step" - Go back to previous instruction
- "Repeat" - Repeat current step
- "Set timer for 5 minutes" - Start a cooking timer

## 🔧 Configuration

### Environment Variables
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Configuration
- **Recipe Generation**: `gpt-4o-mini` (optimized for speed and cost)
- **Image Analysis**: `gpt-4o` (high accuracy vision model)
- **Speech Recognition**: `whisper-1`
- **Text-to-Speech**: `tts-1`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing powerful AI models
- FastAPI team for the excellent web framework
- LangChain community for LLM integration tools

## 🐛 Known Issues & Limitations

- Requires stable internet connection for AI features
- Voice recognition accuracy may vary with background noise
- Image quality affects ingredient identification accuracy
- Limited to ingredients visible in uploaded images

## 🔮 Future Enhancements

- [ ] Nutritional analysis integration
- [ ] Shopping list generation
- [ ] Recipe rating and favorites
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Integration with smart kitchen appliances

Made with ❤️ and powered by AI
