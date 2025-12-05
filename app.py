import os
import re
import json
import logging
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuration ---
UPLOAD_FOLDER = 'resources'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'epub'}
MAX_FILES = 4

# Ensure resource folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global storage for processed text
knowledge_base = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Text Extraction Helpers ---

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_epub(file_path):
    try:
        book = epub.read_epub(file_path)
        text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text.append(soup.get_text())
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error reading EPUB {file_path}: {e}")
        return ""

def load_file_content(file_path, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    try:
        if ext == 'pdf':
            return extract_text_from_pdf(file_path)
        elif ext == 'epub':
            return extract_text_from_epub(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return ""

def sync_resources_from_disk():
    """Scans the resources folder and populates the knowledge base."""
    logger.info("Syncing resources from disk...")
    current_files = os.listdir(UPLOAD_FOLDER)
    
    # clear current memory to ensure sync
    knowledge_base.clear()
    
    count = 0
    for filename in current_files:
        if allowed_file(filename) and count < MAX_FILES:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            content = load_file_content(file_path, filename)
            if content:
                knowledge_base[filename] = content
                count += 1
                logger.info(f"Loaded {filename} ({len(content)} chars)")

# Initialize resources on startup
sync_resources_from_disk()

def is_persian(text):
    persian_pattern = re.compile(r'[\u0600-\u06FF]')
    return bool(persian_pattern.search(text))

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify_key', methods=['POST'])
def verify_key():
    """Tests the API Key by making a lightweight call."""
    data = request.json
    api_key = data.get('api_key')
    
    if not api_key:
        return jsonify({'valid': False, 'message': 'API Key is missing'}), 400
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Test connection.")
        if response:
            return jsonify({'valid': True, 'message': 'Connection Successful'})
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return jsonify({'valid': False, 'message': str(e)}), 400

@app.route('/get_resources', methods=['GET'])
def get_resources():
    """Returns the current list of loaded resources."""
    if not knowledge_base:
        sync_resources_from_disk()
        
    files = [{'name': name, 'size': len(content)} for name, content in knowledge_base.items()]
    return jsonify({'files': files, 'count': len(files), 'max': MAX_FILES})

@app.route('/upload_resource', methods=['POST'])
def upload_resource():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    sync_resources_from_disk()
    
    if len(knowledge_base) >= MAX_FILES:
        return jsonify({'error': f'Maximum {MAX_FILES} resources allowed. Please delete one via file system or clear all.'}), 403

    if file and allowed_file(file.filename):
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        
        # Reload memory
        sync_resources_from_disk()
        
        return jsonify({
            'message': 'File processed successfully', 
            'filename': filename,
            'count': len(knowledge_base)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/clear_resources', methods=['POST'])
def clear_resources():
    # Physically delete files
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")
    
    knowledge_base.clear()
    return jsonify({'message': 'All resources cleared', 'count': 0})

@app.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    data = request.json
    api_key = data.get('api_key')
    user_prompt = data.get('prompt')

    if not api_key or not user_prompt:
        return jsonify({'error': 'Missing API Key or Prompt'}), 400

    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash', 
            generation_config={"temperature": 0.7}
        )
        
        # PHASE 1: Translation (Persian to English)
        processed_prompt = user_prompt
        was_translated = False
        
        if is_persian(user_prompt):
            translation_instruction = "Translate the following Persian text into SIMPLIFIED, CLARIFIED English. Return only the translation."
            trans_resp = model.generate_content(f"{translation_instruction}\n\n{user_prompt}")
            processed_prompt = trans_resp.text
            was_translated = True

        # PHASE 2: Context Construction
        knowledge_context = ""
        for fname, content in knowledge_base.items():
            knowledge_context += f"\n--- RESOURCE: {fname} ---\n{content[:40000]}...\n"

        # PHASE 3: Generation
        system_instruction = f"""
        You are an Elite Prompt Engineer.
        
        INTERNAL KNOWLEDGE BASE:
        {knowledge_context}
        
        TASK:
        Refine this request: "{processed_prompt}"
        
        Create a highly optimized prompt.
        OUTPUT: Markdown format only.
        """
        
        try:
            response = model.generate_content(
                system_instruction,
                tools='google_search'
            )
        except Exception as tool_error:
            logger.warning(f"Tool usage failed ({tool_error}), retrying without tools.")
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(system_instruction)

        final_output = response.text
        
        meta_info = []
        if was_translated:
            meta_info.append(f"Translated input: '{user_prompt}'")
        if knowledge_context:
            meta_info.append(f"Utilized {len(knowledge_base)} internal resources.")

        return jsonify({
            'result': final_output,
            'meta': meta_info
        })

    except Exception as e:
        logger.error(f"Generation Fatal Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/translate_snippet', methods=['POST'])
def translate_snippet():
    """Translates selected text to simplified Persian."""
    data = request.json
    api_key = data.get('api_key')
    text_to_translate = data.get('text')

    if not api_key or not text_to_translate:
        return jsonify({'error': 'Missing API Key or Text'}), 400

    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_instruction = (
            "As a Professional translator you MUST translate from English To "
            "[SIMPLIFIED] and [CLARIFIED] and [UNDERSTANDABLE] Persian. "
            "Return ONLY the Persian translation."
        )
        
        response = model.generate_content(f"{system_instruction}\n\nText to translate: {text_to_translate}")
        return jsonify({'translation': response.text})

    except Exception as e:
        logger.error(f"Translation Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
