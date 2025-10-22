# Haunted House - Object Oriented Hauntology

An AI-powered chatbot that personifies your house as a caring spirit with knowledge of its history, architecture, and maintenance needs. Using Retrieval Augmented Generation (RAG) with semantic search, the House Spirit draws from historical documents, design specifications, and past conversations to provide contextual, personalized responses.

## Features

- **Personified House Spirit**: Chat with your house as if it were a sentient being with memories and personality
- **Context-Aware Responses**: Semantic search using sentence-transformers embeddings finds the most relevant information
- **Room-Specific Advice**: Get tailored guidance for different areas (kitchen, garden, bedroom, etc.)
- **Streaming Responses**: Real-time response generation for better user experience
- **Conversation History**: All interactions are logged in multiple formats (Markdown, CSV, JSON)
- **Document Knowledge Base**: Automatically processes PDFs, text files, markdown, and images (OCR)
- **Seasonal Awareness**: The house considers seasonal changes in its advice
- **Memory Tracking**: See which documents informed each response with relevance scores

## Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd haunted_house
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   # Optionally customize the Claude model (default: claude-3-5-sonnet-20241022)
   ```

5. **Run the application**
   ```bash
   streamlit run house.py
   ```

The app will open in your browser at `http://localhost:8501`

## Configuration

### House Configuration

Edit `config/house_config.json` to customize your house's details:

```json
{
    "year_built": "1968",
    "architectural_style": "Mid-century Modern",
    "primary_materials": ["stone", "timber", "slate"],
    "rooms": ["living_room", "kitchen", "bedrooms", "bathroom", "garden"],
    "home_systems": ["central_heating", "plumbing", "electrical"],
    "sun_orientation": "south_facing",
    "renovation_history": [
        {"year": 1975, "work": "kitchen_extension"}
    ]
}
```

### House Spirit Personality

Customize the AI's personality by editing `prompts/house_spirit_prompt.txt`

### Model Selection

Configure which Claude model to use in your `.env` file:

```bash
# Choose your Claude model
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # Default: Best balance

# Other options:
# ANTHROPIC_MODEL=claude-3-opus-20240229      # Most capable, slower, more expensive
# ANTHROPIC_MODEL=claude-3-sonnet-20240229    # Balanced performance
# ANTHROPIC_MODEL=claude-3-haiku-20240307     # Fastest, most affordable
```

**Model Comparison:**
- **Claude 3.5 Sonnet** (default): Best choice for most users - excellent intelligence with good speed
- **Claude 3 Opus**: Use when you need maximum reasoning capability for complex questions
- **Claude 3 Sonnet**: Good balance of speed and capability for general use
- **Claude 3 Haiku**: Choose for fastest responses and lower costs on simple queries

### Knowledge Base

Add documents to these directories:
- `documents/` - PDFs, text files, markdown (design docs, manuals, etc.)
- `history/` - Past conversation logs (automatically generated)

Supported formats:
- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)
- Images with text (`.png`, `.jpg`, `.jpeg`) - requires tesseract OCR

## Architecture

### Core Components

1. **RAG System**
   - Uses `sentence-transformers` (all-MiniLM-L6-v2 model) for semantic embeddings
   - Chunks documents using LangChain's CharacterTextSplitter
   - Finds top-3 most relevant chunks via cosine similarity

2. **LLM Integration**
   - Anthropic Claude API with configurable models:
     - Claude 3.5 Sonnet (default) - Best balance of intelligence and speed
     - Claude 3 Opus - Maximum capability for complex reasoning
     - Claude 3 Sonnet - Good balance of speed and intelligence
     - Claude 3 Haiku - Fastest responses for simple queries
   - Supports both streaming and non-streaming modes
   - Custom system prompts configure the house personality

3. **Logging System**
   - **Markdown**: Human-readable conversation history (`history/*.md`)
   - **CSV**: Structured logs with relevance scores (`logs/*.csv`)
   - **JSON**: Machine-readable logs for analysis (`logs/*.json`)

4. **Streamlit UI**
   - Real-time streaming responses
   - Room selection and context awareness
   - Conversation history viewer
   - Audio feedback (ding sound on response)

### Data Flow

```
User Question
    ↓
Semantic Embedding (sentence-transformers)
    ↓
Find Top-3 Relevant Document Chunks (cosine similarity)
    ↓
Build Context + System Prompt
    ↓
Anthropic Claude API (configurable model)
    ↓
Stream Response to UI + Log Everything
```

## Usage Examples

**Ask about maintenance:**
> "What maintenance should I focus on this season?"

**Get historical context:**
> "Tell me about your architectural heritage"

**Room-specific advice:**
> Select "Garden" and ask: "What should I plant this month?"

**Technical questions:**
> "How does my heating system work?"

## Project Structure

```
haunted_house/
├── house.py                 # Main application
├── config/
│   └── house_config.json   # House metadata
├── prompts/
│   └── house_spirit_prompt.txt  # AI personality prompt
├── documents/              # Knowledge base (PDFs, markdown, etc.)
├── history/                # Conversation logs (markdown)
├── logs/                   # Structured logs (CSV, JSON)
├── sounds/
│   └── ding.wav           # Audio feedback
├── images/
│   └── house_spirit.png   # UI image
├── requirements.txt        # Python dependencies
└── .env                   # Environment variables (not in git)
```

## Recent Improvements

- ✅ **Anthropic Claude Integration**: Migrated from Perplexity to Anthropic's Claude models with configurable model selection
- ✅ **Semantic Search**: Upgraded from TF-IDF to sentence-transformers embeddings for better context retrieval
- ✅ **Streaming Responses**: Added real-time response generation with toggle option using Claude's streaming API
- ✅ **Environment Security**: Added `.env` support and comprehensive `.gitignore`
- ✅ **Bug Fixes**: Fixed missing pytesseract import
- ✅ **Documentation**: Comprehensive README with architecture diagrams

## Development

### Adding New Features

1. **Custom document processors**: Extend `load_documents()` in house.py:206
2. **Alternative LLM models**: Modify `get_house_response()` API calls
3. **New room types**: Update `room_options` list and house config
4. **UI customization**: Edit Streamlit components in house.py:580+

### Testing

Run the app locally and test with sample questions:
```bash
streamlit run house.py
```

## Troubleshooting

**"API key not found" error**
- Ensure `.env` file exists with `ANTHROPIC_API_KEY=your_key_here`
- Get your API key from https://console.anthropic.com/

**Slow first response**
- First run downloads the sentence-transformer model (~80MB)
- Subsequent runs use cached model
- Consider using Claude 3 Haiku for faster responses

**OCR not working**
- Install tesseract: `brew install tesseract` (Mac) or `apt-get install tesseract-ocr` (Linux)

**Streaming responses not working**
- Verify your Anthropic API key is valid
- Try disabling streaming with the checkbox
- Check your internet connection

**Rate limit errors**
- Anthropic has rate limits based on your account tier
- Consider upgrading your Anthropic account or using a different model
- Try spacing out your requests

## Credits

- Powered by [Anthropic Claude](https://www.anthropic.com/) AI models
- Built with assistance from [Perplexity AI](https://www.perplexity.ai/)
- Inspired by the concept of "Object Oriented Hauntology"
- House: Hafod, a mid-century modern home in Wolverhampton, UK (1968-1970)

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Areas for improvement:
- Vector database integration (ChromaDB, Pinecone, Weaviate)
- Additional LLM providers (OpenAI GPT-4, local models via Ollama)
- Smart home sensor integration (temperature, humidity, energy usage)
- Voice interface with speech-to-text and text-to-speech
- Mobile app (React Native or Flutter)
- Conversation context across multiple questions in a session
- Analytics dashboard for insights from conversation history

---

*"I am more than walls and timber. I am memory, shelter, and a witness to the lives lived within me."* - The House Spirit
