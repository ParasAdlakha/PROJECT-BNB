import os
import uuid
from google.cloud import firestore, storage
from datetime import datetime

# --- Configuration ---
# ❗️ ❗️ ❗️ EDIT THESE TWO LINES ❗️ ❗️ ❗️
GCP_PROJECT_ID = os.environ.get("my-project-bnb-478507")
STORAGE_BUCKET = os.environ.get("asia-raw-data-bucket-paras")

# Initialize Clients
db = firestore.Client(project=GCP_PROJECT_ID)
storage_client = storage.Client(project=GCP_PROJECT_ID)
bucket = storage_client.bucket(STORAGE_BUCKET)

# --- Firestore Collections ---
RUNS_COLLECTION = db.collection("runs")
SIGNALS_COLLECTION = db.collection("signals")
ANOMALIES_COLLECTION = db.collection("anomalies")
CHAT_LOGS_COLLECTION = db.collection("chat_logs")


def upload_to_gcs(file_data: bytes, run_id: str) -> str:
    """Uploads raw CSV data to Cloud Storage."""
    blob_name = f"raw/{run_id}.csv"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(file_data, content_type='text/csv')
    return f"gs://{STORAGE_BUCKET}/{blob_name}"


def save_run_metadata(run_id: str, metadata: dict):
    """Saves initial run data to the 'runs' collection."""
    data = {
        "timestamp": datetime.now(),
        "status": "processing",
        **metadata
    }
    RUNS_COLLECTION.document(run_id).set(data)


def save_signals(run_id: str, signal_data: list):
    """Saves aggregated signal statistics to the 'signals' collection."""
    batch = db.batch()
    for item in signal_data:
        # Create a unique document ID for each signal within the run
        doc_ref = SIGNALS_COLLECTION.document(f"{run_id}-{item['signal_name']}")
        batch.set(doc_ref, {"run_id": run_id, **item})
    batch.commit()


def save_anomaly_result(run_id: str, anomaly_data: dict):
    """Saves the Gemini-generated anomaly result to the 'anomalies' collection."""
    anomaly_data["timestamp"] = datetime.now()
    ANOMALIES_COLLECTION.document(run_id).set(anomaly_data)
    # Update run status
    RUNS_COLLECTION.document(run_id).update({"status": "completed"})


def fetch_run_data(run_id: str) -> dict:
    """Fetches combined run, signal, and anomaly data for the UI/Chat."""
    run_doc = RUNS_COLLECTION.document(run_id).get()
    if not run_doc.exists:
        return None

    signals_query = SIGNALS_COLLECTION.where("run_id", "==", run_id).stream()
    signals = [doc.to_dict() for doc in signals_query]

    anomaly_doc = ANOMALIES_COLLECTION.document(run_id).get()

    return {
        "run": run_doc.to_dict(),
        "signals": signals,
        "anomaly_result": anomaly_doc.to_dict() if anomaly_doc.exists else None
    }