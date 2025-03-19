from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import numpy as np
import torch
import io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import time
import requests
import markdown2
from PIL import Image

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for data, embeddings, and API keys
data = None
embeddings = {}
GEMINI_API_KEY =  "AIzaSyCcRxsIv0GyiRl3NtPvr1o8LdfoeDUn_HE"
HF_API_KEY = "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"
MODEL_NAME = "gemini-1.5-flash"  # Using Gemini 1.5 Flash model
MODEL = None

# Initialize Gemini API once
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL = genai.GenerativeModel(MODEL_NAME)
    print("Gemini API initialized successfully")
else:
    print("Warning: GEMINI_API_KEY environment variable not set")

# Format markdown response
def format_markdown_response(text):
    """Format the response with proper markdown and styling"""
    # Convert markdown to HTML with extras
    html = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables', 'break-on-newline'])
    
    # Enhance emoji display
    emoji_map = {
        '🏥': '<span class="emoji hospital">🏥</span>',
        '💊': '<span class="emoji medication">💊</span>',
        '⚠️': '<span class="emoji warning">⚠️</span>',
        '📊': '<span class="emoji stats">📊</span>',
        '📋': '<span class="emoji clipboard">📋</span>',
        '👨‍⚕️': '<span class="emoji doctor">👨‍⚕️</span>',
        '🔬': '<span class="emoji research">🔬</span>',
        '📚': '<span class="emoji book">📚</span>',
        '🔍': '<span class="emoji search">🔍</span>',
        '🚨': '<span class="emoji alert">🚨</span>',
        '👁️': '<span class="emoji eye">👁️</span>',
        '🔄': '<span class="emoji repeat">🔄</span>',
        '🔮': '<span class="emoji crystal-ball">🔮</span>'
    }
    
    for emoji, styled_emoji in emoji_map.items():
        html = html.replace(emoji, styled_emoji)
    
    return html

# Image analysis function
def analyze_image(image_data, prompt):
    """Analyzes the image using Gemini 1.5 Flash and returns the response."""
    global MODEL
    try:
        if GEMINI_API_KEY is None:
            return "API key is not configured. Please check environment variables."
        
        if MODEL is None:
            genai.configure(api_key=GEMINI_API_KEY)
            MODEL = genai.GenerativeModel(MODEL_NAME)
        
        image = Image.open(io.BytesIO(image_data))
        
        # Configure the model for image analysis
        generation_config = {
            "temperature": 0.4,  # Lower temperature for more focused medical analysis
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4096,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Create a model instance with our configuration
        configured_model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Generate content with the image and prompt
        response = configured_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {e}"

# Load data from Excel files
def load_excel_data():
    """Load all medical data from Excel files"""
    data = {}
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Read clinical cases
        clinical_df = pd.read_excel('data/clinical_cases.xlsx')
        data['clinical'] = clinical_df.to_dict(orient='records')
        
        # Read medical literature
        literature_df = pd.read_excel('data/medical_literature.xlsx')
        data['literature'] = literature_df.to_dict(orient='records')
        
        # Read symptom cases
        symptom_df = pd.read_excel('data/symptom_cases.xlsx')
        data['symptom'] = symptom_df.to_dict(orient='records')
        
        # Read drug interactions
        drug_df = pd.read_excel('data/drug_interactions.xlsx')
        data['drug'] = drug_df.to_dict(orient='records')
        
        print(f"Data loaded successfully: {sum(len(v) for v in data.values())} total records")
        return data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Generate text for embedding
def prepare_text_for_embedding(record, category):
    """Prepare a record for embedding by converting it to a comprehensive text string"""
    if category == 'clinical':
        text = f"Case ID: {record.get('case_id', '')}. "
        text += f"Patient: {record.get('age', '')} year old {record.get('gender', '')}. "
        text += f"Symptoms: {record.get('symptoms', '')}. "
        text += f"Medical history: {record.get('medical_history', '')}. "
        text += f"Diagnosis: {record.get('diagnosis', '')}. "
        text += f"Treatment: {record.get('treatment', '')}. "
        text += f"Outcome: {record.get('outcome', '')}. "
        text += f"Complications: {record.get('complications', '')}."
    
    elif category == 'literature':
        text = f"Paper ID: {record.get('paper_id', '')}. "
        text += f"Title: {record.get('title', '')}. "
        text += f"Authors: {record.get('authors', '')}. "
        text += f"Published: {record.get('publication_date', '')} in {record.get('journal', '')}. "
        text += f"Key findings: {record.get('key_findings', '')}. "
        text += f"Methodology: {record.get('methodology', '')}. "
        text += f"Sample size: {record.get('sample_size', '')}."
    
    elif category == 'symptom':
        text = f"Symptom ID: {record.get('symptom_id', '')}. "
        text += f"Presenting symptoms: {record.get('presenting_symptoms', '')}. "
        text += f"Diagnosis: {record.get('diagnosis', '')}. "
        text += f"Risk factors: {record.get('risk_factors', '')}. "
        text += f"Specialists: {record.get('recommended_specialists', '')}. "
        text += f"Urgency: {record.get('urgency_level', '')}. "
        text += f"Tests: {record.get('diagnostic_tests', '')}."
    
    elif category == 'drug':
        text = f"Interaction ID: {record.get('interaction_id', '')}. "
        text += f"Medications: {record.get('medications', '')}. "
        text += f"Severity: {record.get('severity', '')}. "
        text += f"Effects: {record.get('effects', '')}. "
        text += f"Recommendations: {record.get('recommendations', '')}. "
        text += f"Alternatives: {record.get('alternatives', '')}."
    
    return text

# Save embeddings to PT file for a specific category
def save_category_embeddings(category_embeddings, category, embeddings_dir='data/embeddings'):
    """Save embeddings for a specific category to a PT file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Create category-specific file path
        file_path = os.path.join(embeddings_dir, f"{category}_embeddings.pt")
        
        # Save embeddings
        torch.save(category_embeddings, file_path)
        print(f"{category.capitalize()} embeddings saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving {category} embeddings: {str(e)}")
        return False

# Load embeddings from PT file for a specific category
def load_category_embeddings(category, embeddings_dir='data/embeddings'):
    """Load embeddings for a specific category from a PT file"""
    try:
        # Create category-specific file path
        file_path = os.path.join(embeddings_dir, f"{category}_embeddings.pt")
        
        if os.path.exists(file_path):
            category_embeddings = torch.load(file_path)
            print(f"{category.capitalize()} embeddings loaded from {file_path}")
            return category_embeddings
        else:
            print(f"{category.capitalize()} embeddings file not found")
            return None
    except Exception as e:
        print(f"Error loading {category} embeddings: {str(e)}")
        return None

# Generate embeddings using HuggingFace Inference API
def get_huggingface_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Get embeddings using HuggingFace Inference API"""
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except Exception as e:
        print(f"Error getting embedding from HuggingFace: {str(e)}")
        return None

# Generate embeddings for a specific category
def generate_category_embeddings(category, records):
    """Generate embeddings for all records in a specific category"""
    try:
        print(f"Generating embeddings for {category} category...")
        
        category_embeddings = []
        
        for record in records:
            # Prepare text representation for embedding
            text = prepare_text_for_embedding(record, category)
            
            # Generate embedding using HuggingFace
            embedding = get_huggingface_embedding(text)
            
            if embedding:
                category_embeddings.append({
                    'record': record,
                    'embedding': embedding
                })
            else:
                id_field = 'case_id' if category == 'clinical' else \
                           'paper_id' if category == 'literature' else \
                           'symptom_id' if category == 'symptom' else \
                           'interaction_id'
                print(f"Warning: Failed to get embedding for record {record.get(id_field, 'ID')}")
        
        print(f"Generated {len(category_embeddings)} embeddings for {category} category")
        return category_embeddings
    
    except Exception as e:
        print(f"Error generating embeddings for {category}: {str(e)}")
        return None

# Calculate cosine similarity
def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

# Find similar records based on query
def find_similar_records(query, category, category_embeddings, top_k=3):
    """Find records most similar to the query in a specific category"""
    # Generate embedding for the query using HuggingFace
    query_embedding = get_huggingface_embedding(query)
    
    if not query_embedding:
        print("Warning: Failed to get embedding for query")
        return []
    
    # Calculate similarities
    similarities = []
    for item in category_embeddings:
        similarity = cosine_similarity(query_embedding, item['embedding'])
        similarities.append({
            'record': item['record'],
            'similarity': similarity
        })
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

# Generate response using Gemini
def generate_gemini_response(query_type, user_query, similar_records):
    """Generate a comprehensive response using Gemini with enhanced formatting"""
    try:
        print("Generating response with Gemini...")
        
        # Format similar records for prompt
        formatted_records = json.dumps([r['record'] for r in similar_records], indent=2)
        similarity_scores = [f"{r['record'].get('case_id' if query_type == 'clinical' else 'paper_id' if query_type == 'literature' else 'symptom_id' if query_type == 'symptom' else 'interaction_id', 'ID')}: {r['similarity']:.2f}" for r in similar_records]
        
        # Create context-specific prompts with enhanced markdown formatting and emojis
        contexts = {
            'clinical': f"""As a medical AI assistant, analyze this case based on similar cases in our database.

User Query: {user_query}

Similar Cases (with similarity scores):
{', '.join(similarity_scores)}

Detailed Case Information:
{formatted_records}

Provide a clinical analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🏥 Case Similarity Analysis

## 💊 Evidence-Based Treatment Recommendations

## ⚠️ Potential Complications to Monitor

## 📊 Expected Outcomes

## 📋 Follow-up Recommendations

For each section, provide detailed medical analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Make sure to maintain clean markdown formatting without extra spaces or characters.""",

            'literature': f"""As a medical research assistant, analyze this research query based on our literature database.

User Query: {user_query}

Relevant Papers (with similarity scores):
{', '.join(similarity_scores)}

Paper Details:
{formatted_records}

Provide a comprehensive literature review with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 📚 Relevant Studies Analysis

## 🔬 Key Findings Synthesis

## 📈 Treatment Efficacy Data

## 📊 Statistical Evidence

## 🔮 Research Gaps & Future Directions

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Make sure to maintain clean markdown formatting without extra spaces or characters.""",

            'symptom': f"""As a diagnostic assistant, analyze these symptoms based on our symptom database.

User Query: {user_query}

Relevant Cases (with similarity scores):
{', '.join(similarity_scores)}

Case Details:
{formatted_records}

Provide a symptom analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🔍 Potential Diagnoses

## ⚠️ Key Risk Factors

## 👨‍⚕️ Specialist Recommendations

## 🚨 Urgency Assessment

## 📋 Recommended Diagnostic Tests

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Make sure to maintain clean markdown formatting without extra spaces or characters.""",

            'drug': f"""As a pharmaceutical expert, analyze these medication interactions.

User Query: {user_query}

Relevant Interactions (with similarity scores):
{', '.join(similarity_scores)}

Interaction Details:
{formatted_records}

Provide a comprehensive interaction analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## ⚠️ Interaction Severity Assessment

## 👁️ Effects to Monitor

## 💊 Medication Adjustments

## 🔄 Alternative Medications

## 📋 Patient Monitoring Guidelines

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Make sure to maintain clean markdown formatting without extra spaces or characters."""
        }
        
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4096,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Use Gemini API
        global MODEL
        if MODEL is None:
            genai.configure(api_key=GEMINI_API_KEY)
            MODEL = genai.GenerativeModel(
                model_name=MODEL_NAME,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        
        # Generate response
        response = MODEL.generate_content(contexts[query_type])
        
        print("Response generated successfully")
        return {
            "status": "success",
            "response": response.text
        }
    
    except Exception as e:
        print(f"Error generating Gemini response: {str(e)}")
        return {
            "status": "error",
            "response": f"Error: {str(e)}"
        }

# Process a medical query end-to-end
def process_medical_query(query_type, user_query, data=None, embeddings=None):
    """Process a medical query end-to-end"""
    try:
        # Load data if not provided
        if data is None:
            data = load_excel_data()
        
        # Get category embeddings
        category_embeddings = embeddings.get(query_type, [])
        if not category_embeddings:
            print(f"Warning: No embeddings found for {query_type} category")
            return {
                "status": "error",
                "response": f"No embeddings available for {query_type} category. Please refresh embeddings."
            }
        
        # Find similar records
        similar_records = find_similar_records(user_query, query_type, category_embeddings)
        
        # Generate response with Gemini
        response = generate_gemini_response(query_type, user_query, similar_records)
        
        return response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            "status": "error",
            "response": f"An error occurred while processing your query: {str(e)}"
        }

# Initialize app data and models
def initialize_app():
    global data, embeddings
    
    try:
        print("Initializing application...")
        
        # Check API keys
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY environment variable not set")
            return False
            
        if not HF_API_KEY:
            print("Warning: HF_API_KEY environment variable not set")
            return False
        
        # Create embeddings directory if it doesn't exist
        embeddings_dir = 'data/embeddings'
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Load data from Excel files
        data = load_excel_data()
        if not data:
            print("Error: Failed to load data")
            return False
        
        # Check and load embeddings for each category
        categories = ['clinical', 'literature', 'symptom', 'drug']
        
        for category in categories:
            print(f"Loading embeddings for {category} category...")
            
            # Try to load existing embeddings
            category_embeddings = load_category_embeddings(category, embeddings_dir)
            
            # If embeddings don't exist, generate and save them
            if category_embeddings is None:
                print(f"Generating new embeddings for {category} category...")
                category_embeddings = generate_category_embeddings(category, data[category])
                
                if category_embeddings:
                    save_category_embeddings(category_embeddings, category, embeddings_dir)
                else:
                    print(f"Error: Failed to generate embeddings for {category} category")
                    return False
            
            # Store the embeddings
            embeddings[category] = category_embeddings
            print(f"Loaded {len(category_embeddings)} embeddings for {category} category")
        
        print("Application initialized successfully")
        return True
    
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    global data, embeddings
    
    # Check if initialization is complete
    if data is None or not embeddings:
        if not initialize_app():
            return jsonify({
                "status": "error",
                "response": "Application failed to initialize properly. Please check logs."
            })
    
    # Get query data from request
    query_data = request.json
    query_type = query_data.get('type')
    user_query = query_data.get('query')
    
    # Ensure valid query type
    valid_types = ['clinical', 'literature', 'symptom', 'drug']
    if query_type not in valid_types:
        return jsonify({
            "status": "error",
            "response": f"Invalid query type. Must be one of: {', '.join(valid_types)}"
        })
    
    # Process the query
    start_time = time.time()
    response = process_medical_query(
        query_type, 
        user_query, 
        data=data, 
        embeddings=embeddings
    )
    processing_time = time.time() - start_time
    
    # Add processing time to response
    response["processing_time"] = f"{processing_time:.2f}s"
    
    return jsonify(response)

@app.route('/analyze-image', methods=['POST'])
def analyze_medical_image():
    try:
        if 'image' not in request.files or 'prompt' not in request.form:
            return jsonify({
                "status": "error",
                "response": "Missing image or prompt"
            })
        
        image_file = request.files['image']
        prompt = request.form['prompt']
        
        if image_file.filename == '':
            return jsonify({
                "status": "error",
                "response": "No image selected"
            })
            
        # Read the image data
        image_data = image_file.read()
        
        # Add medical context to the prompt
        enhanced_prompt = f"""As a medical image analysis expert, please analyze this medical image with precision and clinical relevance. 

Medical Context: {prompt}

Please provide your analysis in a clear, structured format with the following sections:
1. Image Description - Describe what you see in the image
2. Key Findings - Identify notable features or abnormalities
3. Possible Interpretations - Discuss what these findings might indicate
4. Recommendations - Suggest next steps or further tests if applicable

Format your response with clear markdown headings and bullet points for readability."""
        
        # Analyze the image
        start_time = time.time()
        analysis = analyze_image(image_data, enhanced_prompt)
        processing_time = time.time() - start_time
        
        # Format the response in markdown and then to HTML
        html_response = format_markdown_response(analysis)
        
        return jsonify({
            "status": "success",
            "response": html_response,
            "raw_response": analysis,
            "processing_time": f"{processing_time:.2f}s"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "response": f"Error analyzing image: {str(e)}"
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global data, embeddings
    
    # Get count of embeddings for each category
    embedding_counts = {category: len(embs) for category, embs in embeddings.items()}
    
    status = {
        "status": "healthy",
        "data_loaded": data is not None,
        "embeddings_loaded": bool(embeddings),
        "embedding_counts": embedding_counts,
        "gemini_api": GEMINI_API_KEY is not None,
        "huggingface_api": HF_API_KEY is not None
    }
    
    if not data or not embeddings or not GEMINI_API_KEY or not HF_API_KEY:
        status["status"] = "unhealthy"
    
    return jsonify(status)

@app.route('/refresh-embeddings', methods=['POST'])
def refresh_embeddings():
    """Endpoint to regenerate embeddings for all categories or a specific category"""
    global data, embeddings
    
    if not HF_API_KEY:
        return jsonify({
            "status": "error",
            "message": "HF_API_KEY environment variable not set"
        })
    
    try:
        # Get category to refresh (optional)
        request_data = request.json or {}
        category = request_data.get('category')
        
        # Load data if not already loaded
        if data is None:
            data = load_excel_data()
            if not data:
                return jsonify({
                    "status": "error",
                    "message": "Failed to load data"
                })
        
        # Create embeddings directory if it doesn't exist
        embeddings_dir = 'data/embeddings'
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # If category is specified, refresh only that category
        if category:
            if category not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid category: {category}"
                })
                
            start_time = time.time()
            
            # Generate new embeddings
            category_embeddings = generate_category_embeddings(category, data[category])
            
            if not category_embeddings:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to generate embeddings for {category}"
                })
            
            # Save the new embeddings
            save_success = save_category_embeddings(category_embeddings, category, embeddings_dir)
            
            if not save_success:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to save embeddings for {category}"
                })
            
            # Update the global embeddings
            embeddings[category] = category_embeddings
            
            processing_time = time.time() - start_time
            
            return jsonify({
                "status": "success",
                "message": f"Embeddings for {category} refreshed successfully",
                "processing_time": f"{processing_time:.2f}s",
                "embedding_count": len(category_embeddings)
            })
        
        # Otherwise, refresh all categories
        else:
            start_time = time.time()
            results = {}
            
            for category in data.keys():
                # Generate new embeddings
                category_embeddings = generate_category_embeddings(category, data[category])
                
                if not category_embeddings:
                    results[category] = "Failed to generate embeddings"
                    continue
                
                # Save the new embeddings
                save_success = save_category_embeddings(category_embeddings, category, embeddings_dir)
                
                if not save_success:
                    results[category] = "Failed to save embeddings"
                    continue
                
                # Update the global embeddings
                embeddings[category] = category_embeddings
                
                results[category] = f"Success: {len(category_embeddings)} embeddings"
            
            processing_time = time.time() - start_time
            
            return jsonify({
                "status": "success",
                "message": "Embeddings refreshed",
                "processing_time": f"{processing_time:.2f}s",
                "results": results
            })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to refresh embeddings: {str(e)}"
        })

@app.route('/templates/index.html')
def get_index_template():
    return render_template('index.html')

# Create static folder if it doesn't exist
def create_folders():
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

# Call this function directly when the app starts
create_folders()

if __name__ == '__main__':
    # Initialize app before starting Flask
    create_folders()  # Create necessary folders
    initialize_app()  # Initialize the app
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)


