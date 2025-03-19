from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_cors import CORS
import pandas as pd
import os
import numpy as np
import torch
import io
import json
import time
import requests
import markdown2
from PIL import Image
import threading
import logging
import queue
import re
import uuid
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part, Content, GenerationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("mediassist_agent")

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = 'appointment_booking_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'

# Global variables and state tracking
APP_STATE = {
    "is_initialized": False,
    "initialization_error": None,
    "loading_progress": {
        "data": {"status": "pending", "message": "Not started"},
        "embeddings": {"status": "pending", "message": "Not started"},
        "models": {"status": "pending", "message": "Not started"}
    }
}

# Initialize Vertex AI
aiplatform.init(project="ai-heal-cd-care01", location="us-central1")

# Model configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash"
EMBED_MODEL_NAME = "textembedding-gecko"

# Store appointments
appointments = []

def generate_meet_code():
    first = ''.join(random.choices(string.ascii_lowercase, k=3))
    second = ''.join(random.choices(string.ascii_lowercase, k=4))
    third = ''.join(random.choices(string.ascii_lowercase, k=3))
    return f"{first}-{second}-{third}"

def send_appointment_email(appointment, doctor_name, patient_email, meet_link):
    sender_email = "knvdurgaprasad610@gmail.com"
    app_password = "eesc wjrl gaqi whct"
    
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = patient_email
    message['Subject'] = f"Appointment Confirmation with {doctor_name}"
    
    start_time = appointment['start_time']
    end_time = appointment['end_time']
    
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
            <h2 style="color: #3498db; text-align: center;">Your Appointment has been Confirmed</h2>
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p><strong>Doctor:</strong> {doctor_name}</p>
                <p><strong>Date:</strong> {start_time.strftime("%A, %B %d, %Y")}</p>
                <p><strong>Time:</strong> {start_time.strftime("%I:%M %p")} - {end_time.strftime("%I:%M %p")}</p>
                <p><strong>Google Meet Link:</strong> <a href="{meet_link}" style="color: #3498db;">{meet_link}</a></p>
            </div>
            <p>Please join the meeting at the scheduled time using the Google Meet link above.</p>
            <p>Best regards,<br>
            Doctor Appointment Service</p>
        </div>
    </body>
    </html>
    """
    
    message.attach(MIMEText(body, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        
        server.send_message(message)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

class MediAssistAgentSystem:
    def __init__(self):
        self.agents = {}
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.task_queue = queue.Queue()
        self.results_cache = {}
        self.is_initialized = False
        self.embedding_cache = {}

    def initialize(self):
        logger.info("Initializing MediAssist Agent System")
        
        self.model_manager.initialize()
        self.data_manager.initialize()
        self.register_agents()
        
        self.is_initialized = True
        logger.info("MediAssist Agent System initialized successfully")
        return True
        
    def register_agents(self):
        self.agents = {
            "router": RouterAgent(self),
            "clinical": ClinicalAgent(self),
            "literature": LiteratureAgent(self),
            "symptom": SymptomAgent(self),
            "drug": DrugAgent(self),
            "diet": DietAgent(self),
            "image": ImageAgent(self),
            "search": SearchAgent(self),
            "reflection": ReflectionAgent(self)
        }
        logger.info(f"Registered {len(self.agents)} agents")
        
    def process_query(self, query_text: str, query_type: Optional[str] = None, 
                     additional_data: Optional[Dict] = None) -> Dict:
        try:
            logger.info(f"Processing query: '{query_text[:50]}...' (Type: {query_type})")
            
            if not self.is_initialized:
                return {
                    "status": "error",
                    "response": "Agent system is not initialized yet. Please try again in a moment."
                }
            
            task = {
                "query": query_text,
                "type": query_type,
                "additional_data": additional_data or {},
                "status": "pending",
                "start_time": time.time()
            }
            
            if query_type and query_type in self.agents:
                response = self.agents[query_type].process(task)
            else:
                response = self.agents["router"].process(task)
            
            final_response = self.agents["reflection"].process({
                "original_query": query_text,
                "agent_response": response,
                "agent_type": query_type or "auto"
            })
            
            processing_time = time.time() - task["start_time"]
            final_response["processing_time"] = f"{processing_time:.2f}s"
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "response": f"An error occurred while processing your query: {str(e)}"
            }
    
    def process_image_analysis(self, image_data: bytes, prompt: str) -> Dict:
        try:
            logger.info(f"Processing image analysis with prompt: '{prompt[:50]}...'")
            
            if not self.is_initialized:
                return {
                    "status": "error",
                    "response": "Agent system is not initialized yet. Please try again in a moment."
                }
            
            task = {
                "image_data": image_data,
                "prompt": prompt,
                "status": "pending",
                "start_time": time.time()
            }
            
            response = self.agents["image"].process(task)
            
            processing_time = time.time() - task["start_time"]
            response["processing_time"] = f"{processing_time:.2f}s"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing image analysis: {str(e)}")
            return {
                "status": "error",
                "response": f"An error occurred while analyzing your image: {str(e)}"
            }
    
    def process_diet_plan(self, user_data: Dict) -> Dict:
        try:
            logger.info(f"Processing diet plan for user profile")
            
            if not self.is_initialized:
                return {
                    "status": "error",
                    "response": "Agent system is not initialized yet. Please try again in a moment."
                }
            
            task = {
                "user_data": user_data,
                "status": "pending",
                "start_time": time.time()
            }
            
            response = self.agents["diet"].process(task)
            
            processing_time = time.time() - task["start_time"]
            response["processing_time"] = f"{processing_time:.2f}s"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating diet plan: {str(e)}")
            return {
                "status": "error",
                "response": f"An error occurred while generating your diet plan: {str(e)}"
            }
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = self.model_manager.get_embedding(text)
        
        if embedding:
            self.embedding_cache[cache_key] = embedding
            
        return embedding
        
    def refresh_embeddings(self, category: str) -> Dict:
        try:
            logger.info(f"Refreshing embeddings for {category} category")
            
            if not self.data_manager.category_exists(category):
                return {
                    "status": "error",
                    "message": f"Invalid category: {category}"
                }
                
            start_time = time.time()
            result = self.data_manager.refresh_embeddings(category)
            
            if not result["success"]:
                return {
                    "status": "error",
                    "message": result["message"]
                }
                
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": f"Embeddings for {category} refreshed successfully",
                "processing_time": f"{processing_time:.2f}s",
                "embedding_count": result["count"]
            }
            
        except Exception as e:
            logger.error(f"Error refreshing embeddings: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to refresh embeddings: {str(e)}"
            }
    
    def generate_specialists(self, symptoms: str) -> Dict:
        try:
            logger.info(f"Generating specialist recommendations for: '{symptoms[:50]}...'")
            
            specialist_prompt = f"""Based on the following symptoms, generate recommendations for 5 medical specialists in Hyderabad, India who would be most appropriate to consult. For each specialist, provide their name, specialty, hospital/clinic affiliation (must be a hospital or clinic in Hyderabad, India), years of experience, and a brief expert bio (1-2 sentences) images should be from indian.

Symptoms: {symptoms}

Format the response as a structured JSON array of 5 specialists with the following fields:
- id (integer from 1-5)
- name (doctor's full name with title, use Indian names)
- specialty (their medical specialty)
- hospital (where they practice in Hyderabad, India - use real hospital names)
- experience (years of experience as a number)
- bio (brief professional description)
- image_url (leave as "https://randomuser.me/api/portraits/men/[ID].jpg" where [ID] is between 1-99)

Assign each doctor a unique ID (1-5), and ensure the doctors are truly appropriate for the symptoms described, with relevant specialties. The response should ONLY contain the JSON array, no other text.
"""
            
            model = self.model_manager.get_gemini_model()
            
            response = model.generate_content(specialist_prompt)
            
            try:
                doctors_text = response.text
                if "```json" in doctors_text:
                    doctors_text = doctors_text.split("```json")[1].split("```")[0].strip()
                elif "```" in doctors_text:
                    doctors_text = doctors_text.split("```")[1].split("```")[0].strip()
                
                doctors = json.loads(doctors_text)
                
                doctors_with_slots = self.generate_time_slots_for_doctors(doctors)
                
                return {
                    "status": "success",
                    "specialists": doctors_with_slots
                }
            except Exception as e:
                logger.error(f"Error parsing specialist response: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to parse specialist recommendations: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error generating specialists: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate specialist recommendations: {str(e)}"
            }
    
    def generate_time_slots_for_doctors(self, doctors: List[Dict]) -> List[Dict]:
        for doctor in doctors:
            time_slots = []
            doctor_id = doctor["id"]
            
            for day in range(7):
                date = datetime.now() + timedelta(days=day)
                for hour in range(9, 16):
                    start_time = datetime(date.year, date.month, date.day, hour, 0)
                    end_time = start_time + timedelta(minutes=45)
                    time_slots.append({
                        "doctor_id": doctor_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "formatted_time": start_time.strftime("%A, %B %d at %I:%M %p"),
                        "is_booked": False
                    })
            
            doctor["available_slots"] = time_slots
        
        return doctors

class Agent:
    def __init__(self, system):
        self.system = system
        self.name = "base_agent"
        
    def process(self, task: Dict) -> Dict:
        raise NotImplementedError("Each agent must implement its own process method")
    
    def _create_base_response(self, status="success") -> Dict:
        return {
            "status": status,
            "response": "",
            "agent": self.name,
            "timestamp": time.time()
        }
        
    def _generate_direct_response(self, query, category):
        try:
            prompt_templates = {
                'clinical': f"""As a medical AI assistant, provide a detailed clinical analysis for this query:

Query: {query}

Provide a comprehensive clinical analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🏥 Clinical Analysis

## 💊 Treatment Considerations

## ⚠️ Potential Complications

## 📊 Expected Outcomes

## 📋 Recommendations

For each section, provide detailed and evidence-based medical analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms.""",

                'literature': f"""As a medical research assistant, provide a literature-based analysis for this query:

Query: {query}

Provide a comprehensive research perspective with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 📚 Research Overview

## 🔬 Key Scientific Findings

## 📈 Evidence Summary

## 📊 Clinical Implications

## 🔮 Future Research Directions

For each section, provide detailed analysis based on current medical literature. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms.""",

                'symptom': f"""As a diagnostic assistant, analyze these symptoms:

Query: {query}

Provide a symptom analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🔍 Possible Causes

## ⚠️ Important Considerations

## 👨‍⚕️ Medical Advice

## 🚨 When to Seek Help

## 📋 Next Steps

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms.""",

                'drug': f"""As a pharmaceutical expert, provide information about this medication query:

Query: {query}

Provide a comprehensive medication analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 💊 Medication Information

## ⚠️ Important Considerations

## 👁️ Side Effects

## 🔄 Interactions

## 📋 Usage Guidelines

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms.""",

                'diet': f"""As a nutrition expert, provide dietary advice for this query:

Query: {query}

Provide comprehensive nutritional guidance with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🥗 Nutritional Analysis

## ⚖️ Dietary Recommendations

## 🍎 Food Suggestions

## 💧 Hydration Guidelines

## 🏋️ Lifestyle Considerations

For each section, provide detailed and practical advice. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms."""
            }
            
            prompt = prompt_templates.get(category, prompt_templates['clinical'])
            
            model = self.system.model_manager.get_gemini_model()
            
            response = model.generate_content(prompt)
            logger.info(response.text)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating direct response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your query. Please try rephrasing your question or providing more details."

class RouterAgent(Agent):
    def __init__(self, system):
        super().__init__(system)
        self.name = "router"
        
    def process(self, task: Dict) -> Dict:
        try:
            query = task["query"]
            
            if any(word in query.lower() for word in ["symptom", "diagnose", "suffering from", "experiencing", "feeling", "pain", "ache", "headache", "fever", "cough", "rash"]):
                agent_name = "symptom"
                logger.info(f"Router detected symptom keywords, using symptom agent for query: '{query[:50]}...'")
                
                agent_task = task.copy()
                agent_task["type"] = agent_name
                
                response = self.system.agents[agent_name].process(agent_task)
                
                response["agent"] = agent_name
                
                return response
            
            # Define the function for routing
            router_function = FunctionDeclaration(
                name="route_query",
                description="Determine which specialized medical agent should handle this query",
                parameters={
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": ["clinical", "literature", "symptom", "drug", "diet"],
                            "description": "The most appropriate agent for this medical query"
                        }
                    },
                    "required": ["agent"]
                }
            )
            
            tools = [Tool(function_declarations=[router_function])]
            
            model = self.system.model_manager.get_agentic_model()
            
            router_prompt = f"""As an AI medical assistant router, analyze this query and determine which specialized medical agent should handle it:

Query: {query}

Available agents:
1. "clinical" - For clinical cases, patient treatment, medical advice
2. "literature" - For medical research, studies, papers, evidence-based medicine
3. "symptom" - For symptom analysis, diagnoses, medical conditions
4. "drug" - For medication information, drug interactions, pharmaceutical queries
5. "diet" - For nutrition, diet plans, food-related health queries

Choose the single best match from the available agents.
"""

            response = model.generate_content(
                router_prompt,
                tools=tools
            )
            
            function_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call'):
                    function_calls.append(part.function_call)
            
            if function_calls:
                function_call = function_calls[0]
                agent_name = json.loads(function_call.args)["agent"]
            else:
                # Fallback if no function call is detected
                valid_agents = ["clinical", "literature", "symptom", "drug", "diet"]
                potential_agents = [agent for agent in valid_agents if agent in response.text.lower()]
                agent_name = potential_agents[0] if potential_agents else "clinical"
            
            # Validate the agent name
            valid_agents = ["clinical", "literature", "symptom", "drug", "diet"]
            if agent_name not in valid_agents:
                logger.warning(f"Router returned invalid agent: {agent_name}, defaulting to clinical")
                agent_name = "clinical"
                
            logger.info(f"Router selected {agent_name} agent for query: '{query[:50]}...'")
            
            agent_task = task.copy()
            agent_task["type"] = agent_name
            
            response = self.system.agents[agent_name].process(agent_task)
            
            response["agent"] = agent_name
            
            return response
            
        except Exception as e:
            logger.error(f"Error in router agent: {str(e)}")
            response = self._create_base_response("error")
            response["response"] = f"Error determining how to process your query. Please try again."
            return response

class ClinicalAgent(Agent):
    def __init__(self, system):
        super().__init__(system)
        self.name = "clinical"
        
    def process(self, task: Dict) -> Dict:
        try:
            query = task["query"]
            
            similar_cases = self.system.data_manager.find_similar(query, "clinical", top_k=3)
            
            if not similar_cases or len(similar_cases) == 0:
                logger.info(f"No similar clinical cases found for query: '{query[:50]}...', generating direct response")
                
                direct_response = self._generate_direct_response(query, "clinical")
                
                response = self._create_base_response()
                response["response"] = direct_response
                response["direct_response"] = True
                return response
            
            formatted_cases = self._format_similar_cases(similar_cases)
            
            clinical_prompt = f"""As a clinical decision support agent, analyze this medical query based on similar cases in our database.

User Query: {query}

Similar Clinical Cases:
{formatted_cases}

Provide a comprehensive clinical analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🏥 Case Similarity Analysis

## 💊 Evidence-Based Treatment Recommendations

## ⚠️ Potential Complications to Monitor

## 📊 Expected Outcomes

## 📋 Follow-up Recommendations

For each section, provide detailed medical analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative."""

            model = self.system.model_manager.get_gemini_model()
            model_response = model.generate_content(clinical_prompt)
            
            response = self._create_base_response()
            response["response"] = model_response.text
            return response
            
        except Exception as e:
            logger.error(f"Error in clinical agent: {str(e)}")
            response = self._create_base_response("error")
            response["response"] = f"Error processing your clinical query. Please try again."
            return response
    
    def _format_similar_cases(self, cases: List[Dict]) -> str:
        formatted = ""
        for i, case in enumerate(cases):
            record = case["record"]
            formatted += f"Case {i+1} (Similarity: {case['similarity']:.2f}):\n"
            formatted += f"- Age: {record.get('age', 'N/A')}, Gender: {record.get('gender', 'N/A')}\n"
            formatted += f"- Symptoms: {record.get('symptoms', 'N/A')}\n"
            formatted += f"- Diagnosis: {record.get('diagnosis', 'N/A')}\n"
            formatted += f"- Treatment: {record.get('treatment', 'N/A')}\n"
            formatted += f"- Outcome: {record.get('outcome', 'N/A')}\n\n"
        return formatted

class LiteratureAgent(Agent):
    def __init__(self, system):
        super().__init__(system)
        self.name = "literature"
        
    def process(self, task: Dict) -> Dict:
        try:
            query = task["query"]
            
            similar_literature = self.system.data_manager.find_similar(query, "literature", top_k=3)
            
            if not similar_literature or len(similar_literature) == 0:
                logger.info(f"No similar literature found for query: '{query[:50]}...', generating direct response")
                
                direct_response = self._generate_direct_response(query, "literature")
                
                response = self._create_base_response()
                response["response"] = direct_response
                response["direct_response"] = True
                return response
            
            formatted_literature = self._format_similar_literature(similar_literature)
            
            literature_prompt = f"""As a medical research assistant, analyze this research query based on our literature database.

User Query: {query}

Relevant Literature:
{formatted_literature}

Provide a comprehensive literature review with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 📚 Relevant Studies Analysis

## 🔬 Key Findings Synthesis

## 📈 Treatment Efficacy Data

## 📊 Statistical Evidence

## 🔮 Research Gaps & Future Directions

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative."""

            model = self.system.model_manager.get_gemini_model()
            model_response = model.generate_content(literature_prompt)
            
            response = self._create_base_response()
            response["response"] = model_response.text
            return response
            
        except Exception as e:
            logger.error(f"Error in literature agent: {str(e)}")
            response = self._create_base_response("error")
            response["response"] = f"Error processing your literature query. Please try again."
            return response
    
    def _format_similar_literature(self, literature: List[Dict]) -> str:
        formatted = ""
        for i, paper in enumerate(literature):
            record = paper["record"]
            formatted += f"Paper {i+1} (Similarity: {paper['similarity']:.2f}):\n"
            formatted += f"- Title: {record.get('title', 'N/A')}\n"
            formatted += f"- Authors: {record.get('authors', 'N/A')}\n"
            formatted += f"- Journal: {record.get('journal', 'N/A')}, Date: {record.get('publication_date', 'N/A')}\n"
            formatted += f"- Key Findings: {record.get('key_findings', 'N/A')}\n"
            formatted += f"- Methodology: {record.get('methodology', 'N/A')}\n\n"
        return formatted

class SymptomAgent(Agent):
    def __init__(self, system):
        super().__init__(system)
        self.name = "symptom"
        
    def process(self, task: Dict) -> Dict:
        try:
            query = task["query"]
            
            similar_symptoms = self.system.data_manager.find_similar(query, "symptom", top_k=3)
            
            if not similar_symptoms or len(similar_symptoms) == 0:
                logger.info(f"No similar symptom information found for query: '{query[:50]}...', generating direct response")
                
                direct_response = self._generate_direct_response(query, "symptom")
                
                specialists_result = self.system.generate_specialists(query)
                
                response = self._create_base_response()
                response["response"] = direct_response
                response["direct_response"] = True
                response["show_booking"] = True
                
                if specialists_result["status"] == "success":
                    response["specialists"] = specialists_result["specialists"]
                
                return response
            
            formatted_symptoms = self._format_similar_symptoms(similar_symptoms)
            
            symptom_prompt = f"""As a diagnostic assistant, analyze these symptoms based on our symptom database.

User Query: {query}

Relevant Symptom Cases:
{formatted_symptoms}

Provide a symptom analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🔍 Potential Diagnoses

## ⚠️ Key Risk Factors

## 👨‍⚕️ Specialist Recommendations

## 🚨 Urgency Assessment

## 📋 Recommended Diagnostic Tests

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative."""

            model = self.system.model_manager.get_gemini_model()
            model_response = model.generate_content(symptom_prompt)
            
            specialists_result = self.system.generate_specialists(query)
            
            response = self._create_base_response()
            response["response"] = model_response.text
            response["show_booking"] = True
            
            if specialists_result["status"] == "success":
                response["specialists"] = specialists_result["specialists"]
            
            return response
            
        except Exception as e:
            logger.error(f"Error in symptom agent: {str(e)}")
            response = self._create_base_response("error")
            response["response"] = f"Error processing your symptom query. Please try again."
            return response
    
    def _format_similar_symptoms(self, symptoms: List[Dict]) -> str:
        formatted = ""
        for i, symptom in enumerate(symptoms):
            record = symptom["record"]
            formatted += f"Case {i+1} (Similarity: {symptom['similarity']:.2f}):\n"
            formatted += f"- Presenting Symptoms: {record.get('presenting_symptoms', 'N/A')}\n"
            formatted += f"- Diagnosis: {record.get('diagnosis', 'N/A')}\n"
            formatted += f"- Risk Factors: {record.get('risk_factors', 'N/A')}\n"
            formatted += f"- Urgency Level: {record.get('urgency_level', 'N/A')}\n"
            formatted += f"- Recommended Tests: {record.get('diagnostic_tests', 'N/A')}\n\n"
        return formatted

class DrugAgent(Agent):
   def __init__(self, system):
       super().__init__(system)
       self.name = "drug"
       
   def process(self, task: Dict) -> Dict:
       try:
           query = task["query"]
           
           similar_drugs = self.system.data_manager.find_similar(query, "drug", top_k=3)
           
           if not similar_drugs or len(similar_drugs) == 0:
               logger.info(f"No similar medication information found for query: '{query[:50]}...', generating direct response")
               
               direct_response = self._generate_direct_response(query, "drug")
               
               response = self._create_base_response()
               response["response"] = direct_response
               response["direct_response"] = True
               return response
           
           formatted_drugs = self._format_similar_drugs(similar_drugs)
           
           drug_prompt = f"""As a pharmaceutical expert, analyze these medication interactions.

User Query: {query}

Relevant Drug Interactions:
{formatted_drugs}

Provide a comprehensive interaction analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## ⚠️ Interaction Severity Assessment

## 👁️ Effects to Monitor

## 💊 Medication Adjustments

## 🔄 Alternative Medications

## 📋 Patient Monitoring Guidelines

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative."""

           model = self.system.model_manager.get_gemini_model()
           model_response = model.generate_content(drug_prompt)
           
           response = self._create_base_response()
           response["response"] = model_response.text
           return response
           
       except Exception as e:
           logger.error(f"Error in drug agent: {str(e)}")
           response = self._create_base_response("error")
           response["response"] = f"Error processing your medication query. Please try again."
           return response
   
   def _format_similar_drugs(self, drugs: List[Dict]) -> str:
       formatted = ""
       for i, drug in enumerate(drugs):
           record = drug["record"]
           formatted += f"Interaction {i+1} (Similarity: {drug['similarity']:.2f}):\n"
           formatted += f"- Medications: {record.get('medications', 'N/A')}\n"
           formatted += f"- Severity: {record.get('severity', 'N/A')}\n"
           formatted += f"- Effects: {record.get('effects', 'N/A')}\n"
           formatted += f"- Recommendations: {record.get('recommendations', 'N/A')}\n"
           formatted += f"- Alternatives: {record.get('alternatives', 'N/A')}\n\n"
       return formatted

class DietAgent(Agent):
   def __init__(self, system):
       super().__init__(system)
       self.name = "diet"
       
   def process(self, task: Dict) -> Dict:
       try:
           if "user_data" in task:
               return self._generate_diet_plan(task["user_data"])
           
           query = task["query"]
           
           similar_diets = self.system.data_manager.find_similar(query, "diet", top_k=3)
           
           if not similar_diets or len(similar_diets) == 0:
               logger.info(f"No similar nutrition information found for query: '{query[:50]}...', generating direct response")
               
               direct_response = self._generate_direct_response(query, "diet")
               
               response = self._create_base_response()
               response["response"] = direct_response
               response["direct_response"] = True
               return response
           
           formatted_diets = self._format_similar_diets(similar_diets)
           
           diet_prompt = f"""As a nutritionist and diet planning expert, analyze this dietary query.

User Query: {query}

Relevant Diet Information:
{formatted_diets}

Provide a comprehensive dietary analysis with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🥗 Nutritional Assessment

## ⚖️ Dietary Recommendations

## 🍎 Food Suggestions

## 💧 Hydration Guidelines

## 🏋️ Exercise Recommendations

## ⚠️ Dietary Considerations

For each section, provide detailed analysis. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative."""

           model = self.system.model_manager.get_gemini_model()
           model_response = model.generate_content(diet_prompt)
           
           response = self._create_base_response()
           response["response"] = model_response.text
           return response
           
       except Exception as e:
           logger.error(f"Error in diet agent: {str(e)}")
           response = self._create_base_response("error")
           response["response"] = f"Error processing your nutrition query. Please try again."
           return response
   
   def _format_similar_diets(self, diets: List[Dict]) -> str:
       formatted = ""
       for i, diet in enumerate(diets):
           record = diet["record"]
           formatted += f"Diet Plan {i+1} (Similarity: {diet['similarity']:.2f}):\n"
           formatted += f"- Age Group: {record.get('age_group', 'N/A')}\n"
           formatted += f"- Weight Range: {record.get('weight_range', 'N/A')}\n"
           formatted += f"- Height Range: {record.get('height_range', 'N/A')}\n"
           formatted += f"- Health Goals: {record.get('health_goals', 'N/A')}\n"
           formatted += f"- Dietary Preferences: {record.get('dietary_preferences', 'N/A')}\n"
           formatted += f"- Recommended Foods: {record.get('recommended_foods', 'N/A')}\n\n"
       return formatted
   
   def _generate_diet_plan(self, user_data: Dict) -> Dict:
       try:
           age = user_data.get('age', '')
           gender = user_data.get('gender', '')
           height = user_data.get('height', '')
           weight = user_data.get('weight', '')
           health_goal = user_data.get('health_goal', '')
           activity_level = user_data.get('activity_level', '')
           dietary_preferences = user_data.get('dietary_preferences', '')
           medical_conditions = user_data.get('medical_conditions', '')
           allergies = user_data.get('allergies', '')
           supplements = user_data.get('supplements', '')
           
           diet_plan_prompt = f"""As a professional nutritionist and dietitian, create a personalized 7-day diet plan for this individual.

User Profile:
- Age: {age}
- Gender: {gender}
- Height: {height}
- Weight: {weight}
- Health Goal: {health_goal}
- Activity Level: {activity_level}
- Dietary Preferences: {dietary_preferences}
- Medical Conditions: {medical_conditions}
- Allergies/Intolerances: {allergies}
- Supplements: {supplements}

Provide a comprehensive nutritional plan with CLEAN markdown formatting. Each section should start with the exact headings below with emojis:

## 🥗 Nutritional Assessment

## ⚖️ Caloric and Macronutrient Recommendations

## 💧 Hydration Guidelines

## 7-Day Meal Plan
### 🍎 Day 1
### 🍎 Day 2
### 🍎 Day 3
### 🍎 Day 4
### 🍎 Day 5
### 🍎 Day 6
### 🍎 Day 7

## 🏋️ Exercise Recommendations

## 📋 Grocery List

## ⚠️ Special Considerations

For each section, provide detailed, practical guidance tailored to this individual. Format your response with bullet points using * (not -), use numbered lists (1. 2.) where appropriate, and use **bold text** for important information or terms. Keep your response concise but informative.

Provide SPECIFIC MEAL IDEAS for each day that are practical, accessible, and aligned with the dietary preferences specified. Include approximate portion sizes where appropriate.

For the grocery list, organize items by category (proteins, fruits, vegetables, grains, etc.) to make shopping easier.
"""
           
           model = self.system.model_manager.get_gemini_model()
           
           response = model.generate_content(diet_plan_prompt)
           
           result = self._create_base_response()
           result["response"] = response.text
           return result
           
       except Exception as e:
           logger.error(f"Error generating diet plan: {str(e)}")
           response = self._create_base_response("error")
           response["response"] = f"Error generating your diet plan. Please try again."
           return response

class ImageAgent(Agent):
   def __init__(self, system):
       super().__init__(system)
       self.name = "image"
       
   def process(self, task: Dict) -> Dict:
       try:
           image_data = task["image_data"]
           prompt = task["prompt"]
           
           import hashlib
           img_hash = hashlib.md5(image_data).hexdigest()
           prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
           cache_key = f"{img_hash}_{prompt_hash}"
           
           if cache_key in self.system.results_cache:
               logger.info(f"Using cached image analysis result for {cache_key}")
               cached_result = self.system.results_cache[cache_key]
               response = self._create_base_response()
               response["response"] = cached_result
               response["cached"] = True
               return response
           
           image = Image.open(io.BytesIO(image_data))
           
           enhanced_prompt = f"""As a medical image analysis expert, analyze this medical image with precision and clinical relevance.

Medical Context: {prompt}

Please provide your analysis in a clear, structured format with the following sections:
1. Image Description - Describe what you see in the image
2. Key Findings - Identify notable features or abnormalities
3. Possible Interpretations - Discuss what these findings might indicate
4. Recommendations - Suggest next steps or further tests if applicable

Format your response with clear markdown headings and bullet points for readability."""
           
           model = self.system.model_manager.get_gemini_model()
           
           # Convert PIL Image to bytes for Vertex AI
           imgByteArr = io.BytesIO()
           image.save(imgByteArr, format=image.format if image.format else 'JPEG')
           image_bytes = imgByteArr.getvalue()
           
           # Create multimodal content parts
           image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
           text_part = Part.from_text(enhanced_prompt)
           multimodal_content = Content(role="user", parts=[text_part, image_part])
           
           response_content = model.generate_content(multimodal_content)
           response_text = response_content.text
           
           self.system.results_cache[cache_key] = response_text
           
           response = self._create_base_response()
           response["response"] = response_text
           return response
           
       except Exception as e:
           logger.error(f"Error in image agent: {str(e)}")
           response = self._create_base_response("error")
           response["response"] = f"Error analyzing your medical image. Please try again."
           return response

class SearchAgent(Agent):
   def __init__(self, system):
       super().__init__(system)
       self.name = "search"
       
   def process(self, task: Dict) -> Dict:
       try:
           query = task.get("query", "")
           
           response = self._create_base_response()
           response["response"] = f"I've searched for additional information on '{query}', but external search capabilities are not currently enabled. I'll provide information based on my existing knowledge."
           
           return response
           
       except Exception as e:
           logger.error(f"Error in search agent: {str(e)}")
           response = self._create_base_response("error")
           response["response"] = f"Error performing search. Please try again."
           return response

class ReflectionAgent(Agent):
   def __init__(self, system):
       super().__init__(system)
       self.name = "reflection"
       
   def process(self, task: Dict) -> Dict:
       try:
           original_query = task.get("original_query", "")
           agent_response = task.get("agent_response", {})
           agent_type = task.get("agent_type", "unknown")
           
           if agent_response.get("status") != "success":
               return agent_response
           
           response_text = agent_response.get("response", "")
           
           if len(response_text) < 100:
               logger.warning(f"Response from {agent_type} agent seems too short, adding disclaimer")
               response_text += "\n\n*Note: This response is limited. For more comprehensive information, please provide additional details or consult a healthcare professional.*"
           
           if any(keyword in original_query.lower() for keyword in ["diagnosis", "treatment", "cure", "remedy"]):
               if "professional" not in response_text.lower() and "consult" not in response_text.lower():
                   response_text += "\n\n*Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for medical concerns.*"
           
           agent_response["response"] = response_text
           return agent_response
           
       except Exception as e:
           logger.error(f"Error in reflection agent: {str(e)}")
           return task.get("agent_response", {
               "status": "error",
               "response": "Error processing your request."
           })

class DataManager:
   def __init__(self):
       self.data = {}
       self.embeddings = {}
       self.data_dir = "data"
       self.embeddings_dir = "data/embeddings"
       self.data_lock = threading.RLock()
       self.embedding_lock = threading.RLock()
       
   def initialize(self):
       logger.info("Initializing Data Manager")
       
       os.makedirs(self.data_dir, exist_ok=True)
       os.makedirs(self.embeddings_dir, exist_ok=True)
       
       APP_STATE["loading_progress"]["data"] = {"status": "in_progress", "message": "Loading data files"}
       self.load_data()
       
       APP_STATE["loading_progress"]["embeddings"] = {"status": "in_progress", "message": "Loading embeddings"}
       self.load_embeddings()
       
       logger.info("Data Manager initialized successfully")
       return True
   
   def load_data(self):
       try:
           categories = [
               ('clinical', 'clinical_cases.xlsx'),
               ('literature', 'medical_literature.xlsx'),
               ('symptom', 'symptom_cases.xlsx'),
               ('drug', 'drug_interactions.xlsx'),
               ('diet', 'diet_plans.xlsx')
           ]
           
           success = True
           for category, filename in categories:
               try:
                   df = pd.read_excel(f'{self.data_dir}/{filename}')
                   self.data[category] = df.to_dict(orient='records')
                   logger.info(f"Loaded {len(self.data[category])} {category} records")
               except Exception as e:
                   logger.error(f"Error loading {category} data: {str(e)}")
                   success = False
           
           if success:
               APP_STATE["loading_progress"]["data"] = {"status": "complete", "message": f"Loaded {sum(len(v) for v in self.data.values())} total records"}
           else:
               APP_STATE["loading_progress"]["data"] = {"status": "warning", "message": "Some data files could not be loaded"}
               
       except Exception as e:
           logger.error(f"Error loading data: {str(e)}")
           APP_STATE["loading_progress"]["data"] = {"status": "error", "message": str(e)}
   
   def load_embeddings(self):
       try:
           categories = ['clinical', 'literature', 'symptom', 'drug', 'diet']
           
           success = True
           for category in categories:
               try:
                   embedding_path = f"{self.embeddings_dir}/{category}_embeddings.pt"
                   if os.path.exists(embedding_path):
                       self.embeddings[category] = torch.load(embedding_path, map_location=torch.device('cpu'))
                       logger.info(f"Loaded {len(self.embeddings[category])} {category} embeddings")
                   else:
                       logger.info(f"No embeddings found for {category}, will generate when needed")
               except Exception as e:
                   logger.error(f"Error loading {category} embeddings: {str(e)}")
                   success = False
           
           if success:
               APP_STATE["loading_progress"]["embeddings"] = {"status": "complete", "message": "Embeddings loaded successfully"}
           else:
               APP_STATE["loading_progress"]["embeddings"] = {"status": "warning", "message": "Some embeddings could not be loaded"}
               
       except Exception as e:
           logger.error(f"Error loading embeddings: {str(e)}")
           APP_STATE["loading_progress"]["embeddings"] = {"status": "error", "message": str(e)}
   
   def category_exists(self, category: str) -> bool:
       return category in self.data
   
   def find_similar(self, query: str, category: str, top_k: int = 3) -> List[Dict]:
       try:
           if category not in self.data:
               logger.warning(f"Category {category} not found in data")
               return []
           
           with self.embedding_lock:
               if category not in self.embeddings or not self.embeddings[category]:
                   logger.info(f"Generating embeddings for {category} category")
                   self.generate_embeddings(category)
                   
                   if category not in self.embeddings or not self.embeddings[category]:
                       logger.error(f"Failed to generate embeddings for {category}")
                       return []
           
           model_manager = ModelManager()
           query_embedding = model_manager.get_embedding(query)
           
           if not query_embedding:
               logger.warning("Failed to get embedding for query")
               return []
           
           similarities = []
           query_embedding_np = np.array(query_embedding)
           
           for item in self.embeddings[category]:
               try:
                   if 'embedding' not in item or not item['embedding']:
                       continue
                       
                   item_embedding = np.array(item['embedding'])
                   
                   if np.all(item_embedding == 0):
                       continue
                       
                   similarity = self.cosine_similarity(query_embedding_np, item_embedding)
                   
                   if similarity > 0.3:
                       similarities.append({
                           'record': item['record'],
                           'similarity': similarity
                       })
               except Exception as e:
                   logger.warning(f"Error calculating similarity for an item: {str(e)}")
                   continue
           
           similarities.sort(key=lambda x: x['similarity'], reverse=True)
           
           logger.info(f"Found {len(similarities)} similar records for category {category}, returning top {min(top_k, len(similarities))}")
           
           return similarities[:top_k]
           
       except Exception as e:
           logger.error(f"Error finding similar records: {str(e)}")
           return []
   
   def generate_embeddings(self, category: str) -> bool:
       try:
           if category not in self.data:
               logger.warning(f"Category {category} not found in data")
               return False
           
           model_manager = ModelManager()
           records = self.data[category]
           category_embeddings = []
           
           for record in records:
               text = self.prepare_text_for_embedding(record, category)
               
               embedding = model_manager.get_embedding(text)
               
               if embedding:
                   category_embeddings.append({
                       'record': record,
                       'embedding': embedding
                   })
               else:
                   logger.warning(f"Failed to get embedding for record in {category}")
           
           with self.embedding_lock:
               self.embeddings[category] = category_embeddings
               
               torch.save(category_embeddings, f"{self.embeddings_dir}/{category}_embeddings.pt")
           
           logger.info(f"Generated {len(category_embeddings)} embeddings for {category} category")
           return True
           
       except Exception as e:
           logger.error(f"Error generating embeddings for {category}: {str(e)}")
           return False
   
   def refresh_embeddings(self, category: str) -> Dict:
       try:
           if category not in self.data:
               return {"success": False, "message": f"Category {category} not found in data", "count": 0}
           
           success = self.generate_embeddings(category)
           
           if not success:
               return {"success": False, "message": f"Failed to generate embeddings for {category}", "count": 0}
           
           count = len(self.embeddings[category])
           return {"success": True, "message": f"Generated {count} embeddings for {category}", "count": count}
           
       except Exception as e:
           logger.error(f"Error refreshing embeddings for {category}: {str(e)}")
           return {"success": False, "message": str(e), "count": 0}
   
   def prepare_text_for_embedding(self, record: Dict, category: str) -> str:
       try:
           if category == 'clinical':
               return f"Case ID: {record.get('case_id', '')}. Patient: {record.get('age', '')} year old {record.get('gender', '')}. Symptoms: {record.get('symptoms', '')}. Medical history: {record.get('medical_history', '')}. Diagnosis: {record.get('diagnosis', '')}. Treatment: {record.get('treatment', '')}. Outcome: {record.get('outcome', '')}. Complications: {record.get('complications', '')}."
           
           elif category == 'literature':
               return f"Paper ID: {record.get('paper_id', '')}. Title: {record.get('title', '')}. Authors: {record.get('authors', '')}. Published: {record.get('publication_date', '')} in {record.get('journal', '')}. Key findings: {record.get('key_findings', '')}. Methodology: {record.get('methodology', '')}. Sample size: {record.get('sample_size', '')}."
           
           elif category == 'symptom':
               return f"Symptom ID: {record.get('symptom_id', '')}. Presenting symptoms: {record.get('presenting_symptoms', '')}. Diagnosis: {record.get('diagnosis', '')}. Risk factors: {record.get('risk_factors', '')}. Specialists: {record.get('recommended_specialists', '')}. Urgency: {record.get('urgency_level', '')}. Tests: {record.get('diagnostic_tests', '')}."
           
           elif category == 'drug':
               return f"Interaction ID: {record.get('interaction_id', '')}. Medications: {record.get('medications', '')}. Severity: {record.get('severity', '')}. Effects: {record.get('effects', '')}. Recommendations: {record.get('recommendations', '')}. Alternatives: {record.get('alternatives', '')}."
           
           elif category == 'diet':
               return f"Diet ID: {record.get('diet_id', '')}. Age Group: {record.get('age_group', '')}. Weight Range: {record.get('weight_range', '')}. Height Range: {record.get('height_range', '')}. Health Goals: {record.get('health_goals', '')}. Dietary Preferences: {record.get('dietary_preferences', '')}. Medical Conditions: {record.get('medical_conditions', '')}. Recommended Foods: {record.get('recommended_foods', '')}. Foods to Avoid: {record.get('foods_to_avoid', '')}. Exercise Recommendations: {record.get('exercise_recommendations', '')}."
           
           return ""
       except Exception as e:
           logger.error(f"Error preparing text for embedding: {str(e)}")
           return ""
   
   def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
       try:
           v1 = v1.flatten()
           v2 = v2.flatten()
           
           dot_product = np.dot(v1, v2)
           norm_v1 = np.linalg.norm(v1)
           norm_v2 = np.linalg.norm(v2)
           
           if norm_v1 == 0 or norm_v2 == 0:
               return 0
               
           return float(dot_product / (norm_v1 * norm_v2))
       except Exception as e:
           logger.error(f"Error calculating cosine similarity: {str(e)}")
           return 0

class ModelManager:
   def __init__(self):
       self.embedding_model = None
       self.gemini_model = None
       self.agentic_model = None
       
   def initialize(self):
       logger.info("Initializing Model Manager")
       
       APP_STATE["loading_progress"]["models"] = {"status": "in_progress", "message": "Initializing models"}
       
       try:
           # Initialize Vertex AI text embedding model
           self.embedding_model = EMBED_MODEL_NAME
           
           # Initialize Gemini model
           self.gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
           
           # Initialize agentic model with function calling capability
           self.agentic_model = GenerativeModel(GEMINI_MODEL_NAME)
           
           logger.info(f"Initialized Vertex AI models")
           
           APP_STATE["loading_progress"]["models"] = {"status": "complete", "message": "Models initialized successfully"}
           return True
           
       except Exception as e:
           logger.error(f"Error initializing models: {str(e)}")
           APP_STATE["loading_progress"]["models"] = {"status": "error", "message": str(e)}
           return False
   
   def get_gemini_model(self):
       """Get the Gemini model instance"""
       return self.gemini_model
   
   def get_agentic_model(self):
       """Get the agentic model instance with function calling capability"""
       return self.agentic_model
   
   def get_embedding(self, text: str) -> Optional[List[float]]:
       """Get embedding for text using Vertex AI Embeddings"""
       try:
           # Initialize the embedding model
           embedding_model = aiplatform.TextEmbeddingModel.from_pretrained(self.embedding_model)
           
           # Get embeddings
           embeddings = embedding_model.get_embeddings([text])
           
           if embeddings and len(embeddings) > 0 and embeddings[0].values:
               return embeddings[0].values
           
           return None
           
       except Exception as e:
           logger.error(f"Error getting embedding: {str(e)}")
           return None

def format_markdown_response(text):
   try:
       html = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables', 'break-on-newline'])
       
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
           '🔮': '<span class="emoji crystal-ball">🔮</span>',
           '🥗': '<span class="emoji salad">🥗</span>',
           '🏋️': '<span class="emoji exercise">🏋️</span>',
           '⚖️': '<span class="emoji balance">⚖️</span>',
           '🍎': '<span class="emoji apple">🍎</span>',
           '💧': '<span class="emoji water">💧</span>'
       }
       
       for emoji, styled_emoji in emoji_map.items():
           html = html.replace(emoji, styled_emoji)
       
       return html
   except Exception as e:
       logger.error(f"Error formatting markdown: {str(e)}")
       return f"<p>Error formatting response: {str(e)}</p><pre>{text}</pre>"

agent_system = MediAssistAgentSystem()

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
   return jsonify({
       "initialized": APP_STATE["is_initialized"],
       "error": APP_STATE["initialization_error"],
       "progress": APP_STATE["loading_progress"]
   })

@app.route('/query', methods=['POST'])
def process_query():
   try:
       query_data = request.json
       user_query = query_data.get('query')
       
       if not user_query:
           return jsonify({
               "status": "error",
               "response": "Missing query text"
           })
       
       if not agent_system.is_initialized:
           background_thread = threading.Thread(target=agent_system.initialize)
           background_thread.start()
           
           return jsonify({
               "status": "error",
               "response": "System is initializing. Please try again in a moment."
           })
       
       response = agent_system.process_query(user_query)
       
       if response["status"] == "success":
           html_response = format_markdown_response(response["response"])
           response["response"] = html_response
       
       return jsonify(response)
       
   except Exception as e:
       logger.error(f"Error in query endpoint: {str(e)}")
       return jsonify({
           "status": "error",
           "response": f"An error occurred: {str(e)}"
       })

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
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
       
       image_data = image_file.read()
       
       if not agent_system.is_initialized:
           background_thread = threading.Thread(target=agent_system.initialize)
           background_thread.start()
           
           return jsonify({
               "status": "error",
               "response": "System is initializing. Please try again in a moment."
           })
       
       response = agent_system.process_image_analysis(image_data, prompt)
       
       if response["status"] == "success":
           html_response = format_markdown_response(response["response"])
           response["response"] = html_response
       
       return jsonify(response)
       
   except Exception as e:
       logger.error(f"Error in image analysis endpoint: {str(e)}")
       return jsonify({
           "status": "error",
           "response": f"An error occurred: {str(e)}"
       })

@app.route('/diet-plan', methods=['POST'])
def create_diet_plan():
   try:
       user_data = request.json
       
       required_fields = ['age', 'gender', 'height', 'weight', 'health_goal']
       missing_fields = [field for field in required_fields if not user_data.get(field)]
       
       if missing_fields:
           return jsonify({
               "status": "error",
               "response": f"Missing required fields: {', '.join(missing_fields)}"
           })
       
       if not agent_system.is_initialized:
           background_thread = threading.Thread(target=agent_system.initialize)
           background_thread.start()
           
           return jsonify({
               "status": "error",
               "response": "System is initializing. Please try again in a moment."
           })
       
       response = agent_system.process_diet_plan(user_data)
       
       if response["status"] == "success":
           html_response = format_markdown_response(response["response"])
           response["response"] = html_response
       
       return jsonify(response)
       
   except Exception as e:
       logger.error(f"Error in diet plan endpoint: {str(e)}")
       return jsonify({
           "status": "error",
           "response": f"An error occurred: {str(e)}"
       })

@app.route('/refresh-embeddings', methods=['POST'])
def refresh_embeddings():
   try:
       request_data = request.json or {}
       category = request_data.get('category')
       
       if not category:
           return jsonify({
               "status": "error",
               "message": "Category parameter is required"
           })
       
       if not agent_system.is_initialized:
           background_thread = threading.Thread(target=agent_system.initialize)
           background_thread.start()
           
           return jsonify({
               "status": "error",
               "message": "System is initializing. Please try again in a moment."
           })
       
       response = agent_system.refresh_embeddings(category)
       
       return jsonify(response)
       
   except Exception as e:
       logger.error(f"Error in refresh embeddings endpoint: {str(e)}")
       return jsonify({
           "status": "error",
           "message": f"An error occurred: {str(e)}"
       })

@app.route('/templates/<path:path>')
def send_template(path):
   return send_from_directory('templates', path)

@app.route('/get-doctors', methods=['POST'])
def get_doctors():
   try:
       request_data = request.json
       symptoms = request_data.get('symptoms')
       
       if not symptoms:
           return jsonify({
               "status": "error",
               "message": "Symptoms parameter is required"
           })
       
       if not agent_system.is_initialized:
           return jsonify({
               "status": "error",
               "message": "System is initializing. Please try again in a moment."
           })
       
       specialists_result = agent_system.generate_specialists(symptoms)
       
       return jsonify(specialists_result)
       
   except Exception as e:
       logger.error(f"Error getting doctor recommendations: {str(e)}")
       return jsonify({
           "status": "error",
           "message": f"An error occurred: {str(e)}"
       })

@app.route('/book-appointment', methods=['POST'])
def book_appointment():
   try:
       booking_data = request.json
       
       required_fields = ['doctor_id', 'doctor_name', 'slot_id', 'email']
       missing_fields = [field for field in required_fields if not booking_data.get(field)]
       
       if missing_fields:
           return jsonify({
               "status": "error",
               "message": f"Missing required fields: {', '.join(missing_fields)}"
           })
       
       doctor_id = booking_data['doctor_id']
       doctor_name = booking_data['doctor_name']
       slot_id = booking_data['slot_id']
       email = booking_data['email']
       
       specialists = booking_data.get('specialists', [])
       if not specialists:
           return jsonify({
               "status": "error",
               "message": "No specialists data provided"
           })
       
       selected_doctor = None
       for doctor in specialists:
           if doctor['id'] == doctor_id:
               selected_doctor = doctor
               break
               
       if not selected_doctor:
           return jsonify({
               "status": "error",
               "message": f"Doctor with ID {doctor_id} not found"
           })
           
       selected_slot = None
       for slot in selected_doctor.get('available_slots', []):
           if slot_id == slot.get('formatted_time'):
               selected_slot = slot
               break
               
       if not selected_slot:
           return jsonify({
               "status": "error",
               "message": f"Time slot {slot_id} not found for doctor {doctor_id}"
           })
       
       if selected_slot.get('is_booked', False):
           return jsonify({
               "status": "error",
               "message": "This time slot is already booked. Please select another time."
           })
       
       meet_code = generate_meet_code()
       google_meet_link = f"https://meet.google.com/{meet_code}"
       
       selected_slot['is_booked'] = True
       
       appointment_id = str(uuid.uuid4())
       start_time = selected_slot['start_time']
       end_time = selected_slot['end_time']

       if isinstance(start_time, str):
           try:
               start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
           except ValueError:
               from email.utils import parsedate_to_datetime
               try:
                   start_time = parsedate_to_datetime(start_time)
               except:
                   try:
                       start_time = datetime.strptime(start_time, "%a, %d %b %Y %H:%M:%S GMT")
                   except:
                       start_time = datetime.now()

       if isinstance(end_time, str):
           try:
               end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
           except ValueError:
               from email.utils import parsedate_to_datetime
               try:
                   end_time = parsedate_to_datetime(end_time)
               except:
                   try:
                       end_time = datetime.strptime(end_time, "%a, %d %b %Y %H:%M:%S GMT")
                   except:
                       end_time = start_time + timedelta(minutes=45)

       new_appointment = {
           "id": appointment_id,
           "doctor_id": doctor_id,
           "doctor_name": doctor_name,
           "start_time": start_time,
           "end_time": end_time,
           "formatted_time": selected_slot['formatted_time'],
           "patient_email": email,
           "google_meet_link": google_meet_link
       }
       appointments.append(new_appointment)
       
       email_sent = False
       try:
           email_sent = send_appointment_email(new_appointment, doctor_name, email, google_meet_link)
       except Exception as e:
           logger.error(f"Error sending email: {str(e)}")
       
       return jsonify({
           "status": "success",
           "message": "Appointment booked successfully",
           "appointment": {
               "id": appointment_id,
               "doctor_name": doctor_name,
               "time": selected_slot['formatted_time'],
               "google_meet_link": google_meet_link,
               "email_sent": email_sent
           }
       })
       
   except Exception as e:
       logger.error(f"Error booking appointment: {str(e)}")
       return jsonify({
           "status": "error",
           "message": f"An error occurred: {str(e)}"
       })

def create_folders():
   os.makedirs('static', exist_ok=True)
   os.makedirs('templates', exist_ok=True)
   os.makedirs('data', exist_ok=True)
   os.makedirs('data/embeddings', exist_ok=True)
   os.makedirs('cache', exist_ok=True)
   os.makedirs('cache/image_analysis', exist_ok=True)

def background_initialize():
   global APP_STATE
   
   try:
       create_folders()
       
       success = agent_system.initialize()
       
       if success:
           APP_STATE["is_initialized"] = True
           logger.info("Background initialization completed successfully")
       else:
           APP_STATE["initialization_error"] = "Failed to initialize agent system"
           logger.error("Failed to initialize agent system")
           
   except Exception as e:
       APP_STATE["initialization_error"] = str(e)
       logger.error(f"Error in background initialization: {str(e)}")

background_thread = threading.Thread(target=background_initialize)
background_thread.daemon = True
background_thread.start()

if __name__ == '__main__':
   APP_STATE["start_time"] = time.time()
   
   logger.info("Starting Flask application")
   app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)