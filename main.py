from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uuid
import json
import pandas as pd
# ❗️ TEMPORARY FIX: DELETE THIS AFTER TESTING ❗️
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\LENOVO\Desktop\gcp keys\my-project-bnb-478507-efd763f263d5.json"

# Initialize FastAPI and Gemini Client (using Vertex AI SDK)
app = FastAPI(title="ASIA Backend Agent")
# ... rest of the code ...
from google.cloud import aiplatform
# REMOVE: from google.cloud.aiplatform.generative_models import GenerativeModel, GenerationConfig
from vertexai.generative_models import GenerativeModel, GenerationConfig

from . import gcp_services
from . import data_analysis

# Initialize FastAPI and Gemini Client (using Vertex AI SDK)
app = FastAPI(title="ASIA Backend Agent")
# NOTE: Initialize Vertex AI client outside of the request path
# ❗️ ❗️ ❗️ EDIT THIS LOCATION ❗️ ❗️ ❗️
# Initialize the model variable to None globally
model = None

def initialize_gemini_model():
    global model
    if model is None:
        aiplatform.init(project=gcp_services.GCP_PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-2.5-flash")

# Call the initialization function when the app starts up
@app.on_event("startup")
async def startup_event():
    initialize_gemini_model()


# --- Pydantic Schemas for API Input/Output ---
class RunMetadata(BaseModel):
    aircraft_type: str = "Simulated Aileron Test Rig"
    subsystem: str = "AILERON_LEFT_ACTUATOR"

class AnalysisResult(BaseModel):
    run_id: str
    status: str
    results_url: str

class ChatMessage(BaseModel):
    run_id: str
    message: str

# --- System Prompt for Gemini ---
SYSTEM_PROMPT = """
You are ASIA (Aviation Safety Intelligence Agent), an expert reliability engineer.
Your task is to analyze key performance indicators (KPIs) from aircraft flight control actuator data.
Based on the metrics provided in the User message, you must:
1. Determine the severity: HIGH, MEDIUM, or LOW.
2. Identify the likely physical failure mode (e.g., internal leakage, friction, sensor drift).
3. Provide a clear, concise rationale linking the metrics to the failure mode.
4. Suggest a specific, actionable maintenance recommendation.

Your response MUST be a single, valid JSON object with the following structure:
{
  "severity": "HIGH/MEDIUM/LOW",
  "component": "Aileron Left Actuator",
  "rationale": "...",
  "recommended_action": "..."
}
"""

def generate_analysis_prompt(kpis: list) -> str:
    """Generates the full prompt for Gemini based on computed KPIs."""
    kpi_text = "\n".join([str(k) for k in kpis])

    user_prompt = f"""
    Analyze the following key performance indicators (KPIs) for Actuator Degradation:

    --- KPI DATA ---
    {kpi_text}
    --- END KPI DATA ---

    Generate the required JSON output based on this data. Assume any average lag over 0.1 degrees and any positive pressure trend indicates a problem.
    """
    return user_prompt


# --- CORE ENDPOINTS ---

@app.post("/upload", response_model=AnalysisResult)
async def upload_and_analyze(
    file: UploadFile = File(...),
    metadata: str = Form(..., description="JSON string of RunMetadata")
):
    """
    Ingests CSV data, performs analysis, calls Gemini for explanation,
    and stores results in Firestore.
    """
    try:
        run_id = str(uuid.uuid4())
        file_data = await file.read()
        metadata_obj = RunMetadata.parse_raw(metadata).dict()

        # 1. Store Raw Data & Metadata
        gcs_uri = gcp_services.upload_to_gcs(file_data, run_id)
        gcp_services.save_run_metadata(run_id, metadata_obj)

        # 2. Compute KPIs (Tool Call: compute_anomaly_score)
        kpis = data_analysis.analyze_run(file_data, run_id)
        gcp_services.save_signals(run_id, kpis)

        # 3. Call Gemini (for Explanation & Recommendation)
        prompt = generate_analysis_prompt(kpis)
        
        # --- THIS IS THE CORRECTED BLOCK ---
        # Configure model for JSON output
        generation_config = GenerationConfig(
            response_mime_type="application/json"
        )
        
        response = model.generate_content(
            contents=[prompt],
            generation_config=generation_config,
            system_instruction=SYSTEM_PROMPT
        )
        # --- END OF CORRECTED BLOCK ---
        
        # 4. Store Anomaly Result
        anomaly_data = json.loads(response.text)
        anomaly_data["run_id"] = run_id
        gcp_services.save_anomaly_result(run_id, anomaly_data)

        return {
            "run_id": run_id,
            "status": "completed",
            "results_url": f"/results/{run_id}"
        }

    except Exception as e:
        print(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/results/{run_id}")
async def fetch_results(run_id: str):
    """Fetches combined run, signal, and anomaly data for the UI."""
    data = gcp_services.fetch_run_data(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run ID not found.")
    return data


@app.post("/chat")
async def chat_with_agent(chat_message: ChatMessage):
    """Conversational endpoint to query the agent about a specific run."""

    run_data = gcp_services.fetch_run_data(chat_message.run_id)
    if not run_data or not run_data.get('anomaly_result'):
        raise HTTPException(status_code=400, detail="Run not analyzed or not found.")

    # 2. Construct Conversational Prompt
    kb = json.dumps(run_data['anomaly_result'], indent=2)
    kb_signals = json.dumps(run_data['signals'], indent=2)

    chat_prompt = f"""
    You are ASIA, an expert in aircraft system fault analysis. 
    The original diagnosis for this run was: {kb} 
    The signal metrics were: {kb_signals}

    Based on this context, answer the user's question, explaining the technical rationale in plain English.

    User Question: {chat_message.message}
    """

    # 3. Call Gemini
    response = model.generate_content(
        contents=[chat_prompt],
        system_instruction=f"You are ASIA, the aviation reliability expert. Keep your response helpful and based on the provided data."
    )

    return {"response": response.text}