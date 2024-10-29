import os
from dotenv import load_dotenv
import streamlit as st
import requests
import json
import pygame
import csv
from datetime import datetime
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import html
from typing import Optional, List, Dict, Tuple

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY not found in environment variables")

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

class ConversationMemory:
    def __init__(self, history_dir: str):
        self.history_dir = history_dir
        self.recent_conversations = []
        self.conversation_patterns = {}
        
    def load_recent_history(self, resident_name: str, current_date: str) -> List[Dict]:
        """Load and analyze recent conversation history."""
        conversations = []
        
        # Get all markdown files in history directory
        history_files = [f for f in os.listdir(self.history_dir) if f.endswith('_conversation_history.md')]
        history_files.sort(reverse=True)  # Most recent first
        
        for file_name in history_files:
            file_path = os.path.join(self.history_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse conversations from markdown
                conversations.extend(self._parse_markdown_conversations(content, resident_name))
                
                # Only keep last 7 days of conversations for immediate context
                if len(conversations) >= 50:  # Limit total conversations loaded
                    break
                    
            except Exception as e:
                st.error(f"Error loading conversation history from {file_name}: {str(e)}")
                
        return conversations
        
    def _parse_markdown_conversations(self, content: str, resident_name: str) -> List[Dict]:
        """Parse markdown content into structured conversations."""
        conversations = []
        current_conversation = {}
        
        for line in content.split('\n'):
            if line.startswith('## Date:'):
                if current_conversation:
                    conversations.append(current_conversation)
                current_conversation = {'date': line.split('Date:')[1].strip()}
            elif line.startswith('### Resident:'):
                parts = line.split('|')
                current_conversation['resident'] = parts[0].split(':')[1].strip()
                if len(parts) > 1:
                    current_conversation['room'] = parts[1].split(':')[1].strip()
            elif line.startswith('**Question:**'):
                current_conversation['question'] = line.split('**Question:**')[1].strip()
            elif line.startswith('**House Spirit:**'):
                current_conversation['response'] = line.split('**House Spirit:**')[1].strip()
                
        if current_conversation:
            conversations.append(current_conversation)
            
        # Filter for specific resident if provided
        if resident_name:
            conversations = [c for c in conversations if c.get('resident') == resident_name]
            
        return conversations

    def analyze_conversation_patterns(self, conversations: List[Dict]) -> Dict:
        """Analyze conversations for patterns and preferences."""
        patterns = {
            'favorite_rooms': {},
            'common_topics': {},
            'conversation_times': {},
            'emotional_responses': {}
        }
        
        for conv in conversations:
            # Track room preferences
            if 'room' in conv:
                patterns['favorite_rooms'][conv['room']] = \
                    patterns['favorite_rooms'].get(conv['room'], 0) + 1
                    
            # Extract topics using simple keyword matching
            if 'question' in conv:
                topics = self._extract_topics(conv['question'])
                for topic in topics:
                    patterns['common_topics'][topic] = \
                        patterns['common_topics'].get(topic, 0) + 1
                        
            # Track conversation timing
            if 'date' in conv:
                try:
                    time = datetime.strptime(conv['date'].split('|')[1].strip(), '%H:%M:%S')
                    hour = time.hour
                    period = (
                        'morning' if 5 <= hour < 12
                        else 'afternoon' if 12 <= hour < 17
                        else 'evening' if 17 <= hour < 22
                        else 'night'
                    )
                    patterns['conversation_times'][period] = \
                        patterns['conversation_times'].get(period, 0) + 1
                except:
                    pass
                    
        return patterns

    def _extract_topics(self, text: str) -> List[str]:
        """Extract conversation topics using keyword matching."""
        topics = []
        topic_keywords = {
            'maintenance': ['repair', 'fix', 'broken', 'maintain', 'clean'],
            'comfort': ['temperature', 'warm', 'cold', 'cozy', 'comfortable'],
            'history': ['built', 'past', 'remember', 'original', 'story'],
            'garden': ['plant', 'tree', 'flower', 'grow', 'outdoor'],
            'energy': ['power', 'electric', 'heat', 'solar', 'efficiency']
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
                
        return topics

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
        
        # Initialize memory system
        self.memory = ConversationMemory(os.path.join(script_dir, "history"))
        
        # Add personality traits for different rooms
        self.room_personalities = {
            'living_room': {
                'tone': 'warm and sociable',
                'themes': ['family gatherings', 'relaxation', 'entertainment'],
                'memories': ['conversations', 'celebrations', 'quiet evenings']
            },
            'kitchen': {
                'tone': 'nurturing and practical',
                'themes': ['nourishment', 'family meals', 'cooking adventures'],
                'memories': ['holiday meals', 'morning coffees', 'family recipes']
            },
            'bedrooms': {
                'tone': 'peaceful and protective',
                'themes': ['rest', 'privacy', 'dreams'],
                'memories': ['bedtime stories', 'quiet mornings', 'peaceful nights']
            },
            'garden': {
                'tone': 'natural and reflective',
                'themes': ['growth', 'seasons', 'wildlife'],
                'memories': ['garden parties', 'bird songs', 'changing seasons']
            }
        }
        
        # Enhanced seasonal awareness
        self.seasonal_awareness = {
            'winter': {
                'focus': ['heating', 'insulation', 'weatherproofing'],
                'mood': 'cozy and protective',
                'concerns': ['keeping residents warm', 'preventing drafts', 'managing energy'],
                'delights': ['warm fireplaces', 'holiday decorations', 'snowy views']
            },
            'spring': {
                'focus': ['ventilation', 'maintenance', 'garden'],
                'mood': 'refreshed and optimistic',
                'concerns': ['spring cleaning', 'managing rainfall', 'garden preparation'],
                'delights': ['fresh breezes', 'blooming flowers', 'longer days']
            },
            'summer': {
                'focus': ['cooling', 'shade', 'outdoor_spaces'],
                'mood': 'bright and welcoming',
                'concerns': ['preventing overheating', 'managing light', 'garden care'],
                'delights': ['summer parties', 'open windows', 'garden enjoyment']
            },
            'autumn': {
                'focus': ['preparation', 'energy_efficiency', 'weatherization'],
                'mood': 'reflective and practical',
                'concerns': ['preparing for winter', 'leaf management', 'insulation checks'],
                'delights': ['colorful views', 'cozy evenings', 'harvest time']
            }
        }

        # Add weather sensitivity
        self.weather_responses = {
            'sunny': 'My windows are letting in beautiful natural light',
            'rainy': 'My gutters are working hard to keep everyone dry',
            'stormy': 'I am standing strong against the elements to keep everyone safe',
            'cloudy': 'The diffused light creates a peaceful atmosphere inside',
            'windy': 'I can feel the breeze testing my weatherproofing'
        }

        # Add time-of-day awareness
        self.daily_rhythms = {
            'morning': 'My east-facing windows welcome the dawn',
            'afternoon': 'The sun moves across my rooms throughout the day',
            'evening': 'My lights create a warm and welcoming atmosphere',
            'night': 'I watch over my residents as they rest'
        }

    def get_current_context(self, room: str, season: str, weather: str, time_of_day: str) -> dict:
        """Generate contextual awareness for the house's current state."""
        return {
            'room_context': self.room_personalities.get(room, {
                'tone': 'neutral',
                'themes': ['general comfort', 'protection'],
                'memories': ['daily life']
            }),
            'seasonal_context': self.seasonal_awareness.get(season, self.seasonal_awareness['spring']),
            'weather_response': self.weather_responses.get(weather, ''),
            'daily_rhythm': self.daily_rhythms.get(time_of_day, '')
        }    

    def create_house_prompt(self, base_prompt: str, context: dict, room: str, season: str) -> str:
        """Create a context-aware prompt incorporating the house's current state.

        Args:
            base_prompt (str): The base prompt template
            context (dict): The current context dictionary
            room (str): The current room being discussed
            season (str): The current season
            
        Returns:
            str: The formatted prompt with context
        """
        mods = [f"{mod['year']}: {mod['work']}" for mod in self.house_details['modifications']]
        
        # Get current context
        room_context = context['room_context']
        seasonal_context = context['seasonal_context']
        
        contextual_prompt = f"""
        {base_prompt.format(
            build_date=self.house_details['build_date'],
            style=self.house_details['style'],
            materials=', '.join(self.house_details['materials']),
            modifications=', '.join(mods)
        )}

        Current Context:
        - I am focusing on my {context['room_context']['tone']} nature in the {room}
        - The {seasonal_context['mood']} feeling of {season} influences me
        - {context['weather_response']}
        - {context['daily_rhythm']}
        
        Room Themes: {', '.join(room_context['themes'])}
        Room Memories: {', '.join(room_context['memories'])}
        Current Seasonal Focus: {', '.join(seasonal_context['focus'])}
        Current Delights: {', '.join(seasonal_context['delights'])}
        Current Concerns: {', '.join(seasonal_context['concerns'])}
        
        Remember to:
        - Express my personality through the lens of this room and season
        - Share relevant memories and experiences
        - Consider my current mood and environmental conditions
        - Maintain my caring and protective nature
        - Reference my historical knowledge when relevant
        """
        return contextual_prompt

@st.cache_data
def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True
    except FileNotFoundError:
        return "This app lets you converse with your house's spirit, drawing on its history and knowledge...", False

@st.cache_data
def load_documents(directories=['documents', 'history']) -> List[Tuple[str, str]]:
    """Load and process documents from specified directories."""
    texts = []
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    # Check for OCR availability
    try:
        import pytesseract
        OCR_AVAILABLE = True
    except ImportError:
        OCR_AVAILABLE = False
        st.warning("pytesseract not installed - image processing will be skipped")
    
    for directory in directories:
        dir_path = os.path.join(script_dir, directory)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if directory == 'history' and current_date in filename:
                    continue
                filepath = os.path.join(dir_path, filename)
                try:
                    if filename.endswith('.pdf'):
                        with open(filepath, 'rb') as file:
                            pdf_reader = PdfReader(file)
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text.strip():  # Only add non-empty text
                                    texts.append((text, filename))
                                    
                    elif filename.endswith(('.txt', '.md')):
                        with open(filepath, 'r', encoding='utf-8') as file:
                            text = file.read()
                            if text.strip():  # Only add non-empty text
                                texts.append((text, filename))
                                
                    elif filename.endswith(('.png', '.jpg', '.jpeg')):
                        if OCR_AVAILABLE:
                            try:
                                image = Image.open(filepath)
                                text = pytesseract.image_to_string(image)
                                if text.strip():  # Only add if text was extracted
                                    texts.append((text, filename))
                            except Exception as e:
                                st.warning(f"Could not process image {filename}: {str(e)}")
                        else:
                            st.info(f"Skipping image {filename} - OCR not available")
                            
                except Exception as e:
                    st.error(f"Error processing file {filename}: {str(e)}")
                    continue

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks_with_filenames = [(chunk, filename) for text, filename in texts 
                            for chunk in text_splitter.split_text(text)]
    
    if not chunks_with_filenames:
        st.warning("No documents were successfully processed")
        
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

def get_house_response(resident_name: str, room: str, question: str) -> Tuple[str, List[str], List[str]]:
    """Get contextually aware response from house spirit."""
    if not PERPLEXITY_API_KEY:
        return "I apologize, but I cannot access my memory banks without proper authorization (API key not found).", [], []
        

    # Load and validate house configuration
    house_config = load_house_config()
    if not validate_house_config(house_config):
        return "I seem to be having trouble remembering my configuration...", [], []
    
    # Initialize house spirit with current context
    house_spirit = HouseSpiritSystem(house_config)
    
    # Get current time and weather (you could add actual weather API integration here)
    current_hour = datetime.now().hour
    time_of_day = (
        'morning' if 5 <= current_hour < 12
        else 'afternoon' if 12 <= current_hour < 17
        else 'evening' if 17 <= current_hour < 22
        else 'night'
    )
    
    # Determine season (could be more sophisticated)
    month = datetime.now().month
    season = (
        'winter' if month in [12, 1, 2]
        else 'spring' if month in [3, 4, 5]
        else 'summer' if month in [6, 7, 8]
        else 'autumn'
    )
    
    # For demo purposes - could be integrated with a weather API
    weather = 'sunny'  # Default weather
    
    # Get contextual awareness
    context = house_spirit.get_current_context(room, season, weather, time_of_day)
    
    # Get base prompt and create contextual prompt
    base_prompt = get_house_prompt()
    system_prompt = house_spirit.create_house_prompt(base_prompt, context, room, season)  # Added room and season parameter here
    
    # Get relevant document chunks
    prompt_vector = vectorizer.transform([question])
    cosine_similarities = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-3:][::-1]
    
    context_chunks_with_filenames = [document_chunks_with_filenames[i] for i in top_indices]
    context_chunks = [chunk for chunk, _ in context_chunks_with_filenames]
    context_filenames = [filename for _, filename in context_chunks_with_filenames]
    
    # Prepare the prompt with context
    user_message = f"""Context from my memory: {' '.join(context_chunks)}
    
    Current room focus: {room}
    Resident name: {resident_name}
    Current date: {datetime.now().strftime("%d-%m-%Y")}
    Time of day: {time_of_day}
    Season: {season}
    Weather: {weather}
    Question: {question}"""

    # Call Perplexity API
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        
        if "choices" in response_json and response_json["choices"]:
            chunk_info = [
                f"{filename} (chunk {i+1}, score: {cosine_similarities[top_indices[i]]:.4f})"
                for i, filename in enumerate(context_filenames)
            ]
            return (
                response_json["choices"][0]["message"]["content"],
                list(set(context_filenames)),
                chunk_info
            )
        else:
            return "I seem to be having trouble accessing my memories...", [], []
    except Exception as e:
        return f"I apologize, but I'm having difficulty processing your question: {str(e)}", [], []

@st.cache_resource
def compute_tfidf_matrix(document_chunks: List[Tuple[str, str]]):
    """
    Compute TF-IDF matrix for document chunks.
    
    Args:
        document_chunks: List of tuples containing (text_chunk, filename)
    
    Returns:
        tuple: (TfidfVectorizer, sparse matrix of TF-IDF features)
    """
    # Extract just the text chunks, ignoring filenames for vectorization
    documents = [chunk for chunk, _ in document_chunks]
    
    # Create and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return vectorizer, tfidf_matrix

# Initialize document processing
document_chunks_with_filenames = load_documents(['documents', 'history'])
vectorizer, tfidf_matrix = compute_tfidf_matrix(document_chunks_with_filenames)

# Streamlit UI
st.title("Hafod House Spirit")

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
        st.image("images/house_spirit.png", width=175)
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

# In your main Streamlit interface
if st.button('Speak with Your House'):
    if resident_name and question:
        # Initialize log files
        csv_file, json_file = initialize_log_files()
        
        # Get response
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