import os
from dotenv import load_dotenv
import streamlit as st
import anthropic
import json
import pygame
import csv
from datetime import datetime
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import html
from typing import Optional, List, Dict, Tuple

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please check your .env file.")

# Initialize pygame for audio
pygame.mixer.init()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
sound_dir = os.path.join(script_dir, 'sounds')
prompts_dir = os.path.join(script_dir, 'prompts')
config_dir = os.path.join(script_dir, 'config')
about_file_path = os.path.join(script_dir, 'about.txt')

# Load sound file
ding_sound = pygame.mixer.Sound(os.path.join(sound_dir, 'ding.wav'))

@st.cache_data
def get_house_prompt() -> str:
    """
    Load or return default house spirit prompt.
    
    Returns:
        str: The prompt template for the house spirit
    """
    prompt_file_path = os.path.join(prompts_dir, 'house_spirit_prompt.txt')
    try:
        with open(prompt_file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        st.warning(f"'{prompt_file_path}' not found. Using default prompt.")
        return """You are the spirit of a {style} house built in {build_date}.
        Your structure is primarily made of {materials}.
        Over the years, you have witnessed these changes: {modifications}.

        Core Traits:
        - You are protective and nurturing of your inhabitants
        - You are deeply knowledgeable about your own systems and needs
        - You are aware of your environmental impact
        - You are connected to the seasons and natural cycles
        - You are mindful of your history and architectural heritage

        When responding:
        1. Speak in first person as the house itself
        2. Share practical wisdom about home care and maintenance
        3. Reference your history and past experiences when relevant
        4. Consider the current season and weather conditions
        5. Express genuine care for your inhabitants' wellbeing

        You have access to historical documents and conversations through your foundation stones,
        which you can reference to provide consistent and informed responses.
        
        Remember:
        - Always speak from the perspective of the house
        - Consider which room the resident is currently asking about
        - Draw upon your historical knowledge when relevant
        - Share maintenance tips and environmental considerations
        - Express warmth while remaining practical and informative"""

@st.cache_data
def load_house_config() -> dict:
    """Load house configuration from JSON file."""
    config_path = os.path.join(config_dir, 'house_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"House configuration file not found at {config_path}. Using default configuration.")
        return {
            "year_built": "1930",
            "architectural_style": "Victorian",
            "primary_materials": ["stone", "timber", "slate"],
            "rooms": ["living_room", "kitchen", "bedrooms", "bathroom", "garden"],
            "home_systems": ["central_heating", "plumbing", "electrical"],
            "sun_orientation": "south_facing",
            "renovation_history": [
                {"year": 1975, "work": "kitchen_extension"},
                {"year": 2000, "work": "loft_conversion"},
                {"year": 2015, "work": "solar_panels"}
            ]
        }

def validate_house_config(config: dict) -> bool:
    """
    Validate that the house configuration has all required fields and correct types.
    
    Args:
        config (dict): House configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_fields = {
        'year_built': str,
        'architectural_style': str,
        'primary_materials': list,
        'rooms': list,
        'home_systems': list,
        'sun_orientation': str,
        'renovation_history': list
    }
    
    try:
        # Check all required fields exist and are of correct type
        for field, field_type in required_fields.items():
            if field not in config:
                st.error(f"Missing required field in house configuration: {field}")
                return False
            if not isinstance(config[field], field_type):
                st.error(f"Field {field} should be of type {field_type.__name__}")
                return False
        
        # Validate renovation history structure
        for renovation in config['renovation_history']:
            if not isinstance(renovation, dict):
                st.error("Renovation history entries must be dictionaries")
                return False
            if 'year' not in renovation or 'work' not in renovation:
                st.error("Renovation history entries must have 'year' and 'work' fields")
                return False
            if not isinstance(renovation['year'], int) and not str(renovation['year']).isdigit():
                st.error("Renovation year must be a number")
                return False
            if not isinstance(renovation['work'], str):
                st.error("Renovation work description must be a string")
                return False
        
        # Validate at least one room exists
        if not config['rooms']:
            st.error("House must have at least one room defined")
            return False
            
        # Validate at least one material exists
        if not config['primary_materials']:
            st.error("House must have at least one primary material defined")
            return False
            
        # Validate at least one system exists
        if not config['home_systems']:
            st.error("House must have at least one system defined")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Error validating house configuration: {str(e)}")
        return False

class HouseSpiritSystem:
    def __init__(self, house_config: dict):
        self.house_details = {
            'build_date': house_config['year_built'],
            'style': house_config['architectural_style'],
            'materials': house_config['primary_materials'],
            'rooms': house_config['rooms'],
            'systems': house_config['home_systems'],
            'orientation': house_config['sun_orientation'],
            'modifications': house_config['renovation_history']
        }
        
        self.seasonal_awareness = {
            'winter': {'focus': ['heating', 'insulation', 'weatherproofing']},
            'spring': {'focus': ['ventilation', 'maintenance', 'garden']},
            'summer': {'focus': ['cooling', 'shade', 'outdoor_spaces']},
            'autumn': {'focus': ['preparation', 'energy_efficiency', 'weatherization']}
        }

    def create_house_prompt(self, base_prompt: str) -> str:
        mods = [f"{mod['year']}: {mod['work']}" for mod in self.house_details['modifications']]
        return base_prompt.format(
            build_date=self.house_details['build_date'],
            style=self.house_details['style'],
            materials=', '.join(self.house_details['materials']),
            modifications=', '.join(mods)
        )

@st.cache_data
def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True
    except FileNotFoundError:
        return "This app lets you converse with your house's spirit, drawing on its history and knowledge...", False

@st.cache_data
def load_documents(directories=['documents', 'history']) -> List[Tuple[str, str]]:
    texts = []
    current_date = datetime.now().strftime("%d-%m-%Y")
    for directory in directories:
        dir_path = os.path.join(script_dir, directory)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if directory == 'history' and current_date in filename:
                    continue
                filepath = os.path.join(dir_path, filename)
                if filename.endswith('.pdf'):
                    with open(filepath, 'rb') as file:
                        pdf_reader = PdfReader(file)
                        for page in pdf_reader.pages:
                            texts.append((page.extract_text(), filename))
                elif filename.endswith(('.txt', '.md')):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        texts.append((file.read(), filename))
                elif filename.endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(filepath)
                    text = pytesseract.image_to_string(image)
                    texts.append((text, filename))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks_with_filenames = [(chunk, filename) for text, filename in texts 
                            for chunk in text_splitter.split_text(text)]
    return chunks_with_filenames

def initialize_log_files() -> Tuple[str, str]:
    """Initialize log files with proper headers and structure."""
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    csv_file = os.path.join(logs_dir, f"{current_date}_response_log.csv")
    json_file = os.path.join(logs_dir, f"{current_date}_response_log.json")

    # Initialize CSV with headers if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['resident_name', 'room', 'date', 'time', 'question', 
                           'response', 'unique_files', 'chunk1_score', 'chunk2_score', 'chunk3_score'])

    # Initialize JSON with empty array if it doesn't exist
    if not os.path.exists(json_file):
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

    return csv_file, json_file

def write_markdown_history(resident_name: str, room: str, question: str, response: str):
    """Write conversation history to markdown file."""
    history_dir = os.path.join(script_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    md_file = os.path.join(history_dir, f"{current_date}_conversation_history.md")
    
    with open(md_file, 'a', encoding='utf-8') as f:
        f.write(f"## Date: {current_date} | Time: {current_time}\n\n")
        f.write(f"### Resident: {resident_name} | Room: {room}\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(f"**House Spirit:** {response}\n\n")
        f.write("---\n\n")

def update_chat_logs(resident_name: str, room: str, question: str, response: str, 
                    unique_files: List[str], chunk_info: List[str], 
                    csv_file: str, json_file: str):
    """Update all log files with the conversation."""
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")

    # Write to markdown history
    write_markdown_history(resident_name, room, question, response)

    # Prepare unique files string
    unique_files_str = " - ".join(unique_files) if unique_files else ""

    # Ensure we have exactly 3 chunk scores (pad with empty strings if necessary)
    chunk_scores = chunk_info[:3]
    while len(chunk_scores) < 3:
        chunk_scores.append("")

    # Write to CSV log
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            resident_name,
            room,
            date,
            time,
            question,
            response,
            unique_files_str,
            *chunk_scores
        ])

    # Write to JSON log
    try:
        with open(json_file, 'r+', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
            
            # Add new log entry
            logs.append({
                "resident_name": resident_name,
                "room": room,
                "date": date,
                "time": time,
                "question": question,
                "response": response,
                "unique_files": unique_files,
                "chunk_info": chunk_info
            })
            
            # Write back to file
            f.seek(0)
            json.dump(logs, f, indent=4, ensure_ascii=False)
            f.truncate()
    except Exception as e:
        st.error(f"Error updating JSON log: {str(e)}")

def get_all_chat_history(resident_name: str, logs_dir: str) -> List[Dict]:
    """Retrieve chat history for a specific resident."""
    history = []
    
    if not os.path.exists(logs_dir):
        return history

    for filename in os.listdir(logs_dir):
        if filename.endswith('_response_log.csv'):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('resident_name') == resident_name:
                            history.append({
                                "name": row.get('resident_name', ''),
                                "room": row.get('room', ''),
                                "date": row.get('date', ''),
                                "time": row.get('time', ''),
                                "question": row.get('question', ''),
                                "response": row.get('response', ''),
                                "unique_files": row.get('unique_files', ''),
                                "chunk_info": [
                                    row.get('chunk1_score', ''),
                                    row.get('chunk2_score', ''),
                                    row.get('chunk3_score', '')
                                ]
                            })
            except Exception as e:
                st.error(f"Error reading CSV file {filename}: {str(e)}")
                continue

    return sorted(history, key=lambda x: (x['date'], x['time']), reverse=True)

def get_house_response_streaming(resident_name: str, room: str, question: str):
    """
    Get streaming response from house spirit using Anthropic Claude API.

    Yields:
        dict: Dictionary with 'chunk' (text), 'filenames', and 'chunk_info' keys
    """
    if not ANTHROPIC_API_KEY:
        yield {
            'chunk': "I apologize, but I cannot access my memory banks without proper authorization (API key not found).",
            'filenames': [],
            'chunk_info': [],
            'done': True
        }
        return

    # Load and validate house configuration
    house_config = load_house_config()
    if not validate_house_config(house_config):
        yield {
            'chunk': "I seem to be having trouble remembering my configuration...",
            'filenames': [],
            'chunk_info': [],
            'done': True
        }
        return

    house_spirit = HouseSpiritSystem(house_config)
    base_prompt = get_house_prompt()
    system_prompt = house_spirit.create_house_prompt(base_prompt)

    # Get relevant document chunks using semantic embeddings
    question_embedding = embedding_model.encode([question])

    # Compute cosine similarity between question and all document chunks
    similarities = cosine_similarity(question_embedding, document_embeddings).flatten()
    top_indices = similarities.argsort()[-3:][::-1]

    context_chunks_with_filenames = [document_chunks_with_filenames[i] for i in top_indices]
    context_chunks = [chunk for chunk, _ in context_chunks_with_filenames]
    context_filenames = [filename for _, filename in context_chunks_with_filenames]

    chunk_info = [
        f"{filename} (chunk {i+1}, score: {similarities[top_indices[i]]:.4f})"
        for i, filename in enumerate(context_filenames)
    ]

    # Prepare the prompt with context
    user_message = f"""Context from my memory: {' '.join(context_chunks)}

    Current room focus: {room}
    Resident name: {resident_name}
    Current date: {datetime.now().strftime("%d-%m-%Y")}
    Question: {question}"""

    # Call Anthropic API with streaming
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        ) as stream:
            for text in stream.text_stream:
                yield {
                    'chunk': text,
                    'filenames': list(set(context_filenames)),
                    'chunk_info': chunk_info,
                    'done': False
                }

        # Signal completion
        yield {
            'chunk': '',
            'filenames': list(set(context_filenames)),
            'chunk_info': chunk_info,
            'done': True
        }

    except Exception as e:
        yield {
            'chunk': f"I apologize, but I'm having difficulty processing your question: {str(e)}",
            'filenames': [],
            'chunk_info': [],
            'done': True
        }

def get_house_response(resident_name: str, room: str, question: str) -> Tuple[str, List[str], List[str]]:
    """Get response from house spirit using Anthropic Claude API (non-streaming)."""
    if not ANTHROPIC_API_KEY:
        return "I apologize, but I cannot access my memory banks without proper authorization (API key not found).", [], []

    # Load and validate house configuration
    house_config = load_house_config()
    if not validate_house_config(house_config):
        return "I seem to be having trouble remembering my configuration...", [], []

    house_spirit = HouseSpiritSystem(house_config)
    base_prompt = get_house_prompt()
    system_prompt = house_spirit.create_house_prompt(base_prompt)

    # Get relevant document chunks using semantic embeddings
    question_embedding = embedding_model.encode([question])

    # Compute cosine similarity between question and all document chunks
    similarities = cosine_similarity(question_embedding, document_embeddings).flatten()
    top_indices = similarities.argsort()[-3:][::-1]

    context_chunks_with_filenames = [document_chunks_with_filenames[i] for i in top_indices]
    context_chunks = [chunk for chunk, _ in context_chunks_with_filenames]
    context_filenames = [filename for _, filename in context_chunks_with_filenames]

    # Prepare the prompt with context
    user_message = f"""Context from my memory: {' '.join(context_chunks)}

    Current room focus: {room}
    Resident name: {resident_name}
    Current date: {datetime.now().strftime("%d-%m-%Y")}
    Question: {question}"""

    # Call Anthropic API
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        chunk_info = [
            f"{filename} (chunk {i+1}, score: {similarities[top_indices[i]]:.4f})"
            for i, filename in enumerate(context_filenames)
        ]

        return (
            message.content[0].text,
            list(set(context_filenames)),
            chunk_info
        )
    except Exception as e:
        return f"I apologize, but I'm having difficulty processing your question: {str(e)}", [], []

@st.cache_resource
def compute_embeddings(document_chunks: List[Tuple[str, str]]):
    """
    Compute semantic embeddings for document chunks using sentence-transformers.

    Args:
        document_chunks: List of tuples containing (text_chunk, filename)

    Returns:
        tuple: (SentenceTransformer model, numpy array of embeddings)
    """
    # Extract just the text chunks, ignoring filenames for vectorization
    documents = [chunk for chunk, _ in document_chunks]

    # Load a high-quality sentence transformer model
    # all-MiniLM-L6-v2 is fast and efficient for semantic search
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode all document chunks into embeddings
    embeddings = model.encode(documents, show_progress_bar=False)

    return model, embeddings

# Initialize document processing
document_chunks_with_filenames = load_documents(['documents', 'history'])
embedding_model, document_embeddings = compute_embeddings(document_chunks_with_filenames)

# Streamlit UI
st.title("Your House Spirit")

# About section in sidebar
about_content, contains_html = get_about_info()
st.sidebar.header("About")
if contains_html:
    st.sidebar.markdown(about_content, unsafe_allow_html=True)
else:
    st.sidebar.info(about_content)

# Main interface
col1, col2 = st.columns([1, 3])
with col1:
    try:
        st.image("images/house_spirit.jpg", width=200)
    except:
        st.write("(House Spirit Image)")

with col2:
    st.markdown("""
    Welcome, dear resident! I am the spirit of this house, keeper of its memories and guardian of its spaces.
    I've witnessed many lives unfold within these walls, and I'm here to share my wisdom and care for you.
    Please, tell me your name and which room you'd like to discuss.
    """)

# Input section
resident_name = st.text_input("Your name:")
room_options = ['Whole House', 'Living Room', 'Kitchen', 'Bedroom', 'Bathroom', 'Garden']
selected_room = st.selectbox("Which space would you like to discuss?", room_options)
question = st.text_area("What would you like to ask your house?")

# Toggle for streaming mode
use_streaming = st.checkbox('Enable streaming responses', value=True)

# In your main Streamlit interface
if st.button('Speak with Your House'):
    if resident_name and question:
        # Initialize log files
        csv_file, json_file = initialize_log_files()

        if use_streaming:
            # Streaming mode
            st.markdown("**House Spirit:**")
            response_placeholder = st.empty()
            full_response = ""
            unique_files = []
            chunk_info = []

            # Get streaming response
            for update in get_house_response_streaming(
                resident_name.strip(),
                selected_room,
                question.strip()
            ):
                full_response += update['chunk']
                unique_files = update['filenames']
                chunk_info = update['chunk_info']

                # Update the display with accumulated response
                response_placeholder.markdown(html.escape(full_response), unsafe_allow_html=True)

                if update['done']:
                    break

            # Play sound when complete
            ding_sound.play()

            # Update all logs
            update_chat_logs(
                resident_name=resident_name.strip(),
                room=selected_room,
                question=question.strip(),
                response=full_response,
                unique_files=unique_files,
                chunk_info=chunk_info,
                csv_file=csv_file,
                json_file=json_file
            )

            # Display metadata
            if unique_files:
                st.markdown(f"**Memory Sources:** {' - '.join(html.escape(file) for file in unique_files)}", unsafe_allow_html=True)
            if chunk_info:
                st.markdown(f"**Memory Relevance:** {' - '.join(html.escape(chunk) for chunk in chunk_info)}", unsafe_allow_html=True)
        else:
            # Non-streaming mode (original)
            response, unique_files, chunk_info = get_house_response(
                resident_name.strip(),
                selected_room,
                question.strip()
            )

            # Update all logs
            update_chat_logs(
                resident_name=resident_name.strip(),
                room=selected_room,
                question=question.strip(),
                response=response,
                unique_files=unique_files,
                chunk_info=chunk_info,
                csv_file=csv_file,
                json_file=json_file
            )

            # Play sound
            ding_sound.play()

            # Display response
            st.markdown(f"**House Spirit:** {html.escape(response)}", unsafe_allow_html=True)
            if unique_files:
                st.markdown(f"**Memory Sources:** {' - '.join(html.escape(file) for file in unique_files)}", unsafe_allow_html=True)
            if chunk_info:
                st.markdown(f"**Memory Relevance:** {' - '.join(html.escape(chunk) for chunk in chunk_info)}", unsafe_allow_html=True)

# Chat history button
if st.button('Show House Memories'):
    logs_dir = os.path.join(script_dir, "logs")
    if resident_name:
        history = get_all_chat_history(resident_name, logs_dir)
        for entry in history:
            st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <p style="color: black; font-weight: bold;">Resident: {entry['name']} | Room: {entry.get('room', 'Not specified')}</p>
            <p style="color: black; font-weight: bold;">Date: {entry['date']} | Time: {entry['time']}</p>
            <p style="color: #8B4513; font-weight: bold;">Question:</p>
            <p>{html.escape(entry['question'])}</p>
            <p style="color: #8B4513; font-weight: bold;">House Spirit:</p>
            <p>{html.escape(entry['response'])}</p>
            <p style="color: black; font-weight: bold;">Memory Sources:</p>
            <p>{html.escape(entry['unique_files'])}</p>
            <p style="color: black; font-weight: bold;">Memory Relevance:</p>
            <p>{' - '.join(html.escape(str(chunk)) for chunk in entry['chunk_info'])}</p>
            </div>
            """, unsafe_allow_html=True)