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
- Perplexity API key ([get one here](https://www.perplexity.ai/))

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
   # Edit .env and add your Perplexity API key
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
   - Perplexity API with llama-3.1-70b-instruct model
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
Perplexity API (llama-3.1-70b-instruct)
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

- ✅ **Semantic Search**: Upgraded from TF-IDF to sentence-transformers embeddings for better context retrieval
- ✅ **Streaming Responses**: Added real-time response generation with toggle option
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
- Ensure `.env` file exists with `PERPLEXITY_API_KEY=your_key_here`

**Slow first response**
- First run downloads the sentence-transformer model (~80MB)
- Subsequent runs use cached model

**OCR not working**
- Install tesseract: `brew install tesseract` (Mac) or `apt-get install tesseract-ocr` (Linux)

**Streaming responses not working**
- Check Perplexity API supports streaming
- Try disabling streaming with the checkbox

## Credits

- Built with assistance from [Perplexity AI](https://www.perplexity.ai/)
- Inspired by the concept of "Object Oriented Hauntology"
- House: Hafod, a mid-century modern home in Wolverhampton, UK (1968-1970)

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Areas for improvement:
- Vector database integration (ChromaDB, Pinecone)
- Multi-model support (Claude, GPT-4, local models)
- Smart home sensor integration
- Voice interface
- Mobile app

---

*"I am more than walls and timber. I am memory, shelter, and a witness to the lives lived within me."* - The House Spirit
