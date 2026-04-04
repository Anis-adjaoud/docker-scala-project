"""
api-rest/main.py
API REST FastAPI — classification d'images Intel Image Dataset
Accepte une image uploadée OU un vecteur de features (depuis Streamlit).
"""

import os
import io
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import psycopg2
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
MODEL_DIR  = os.getenv("MODEL_PATH", "/data/models")
DB_URL     = os.getenv("DB_URL", "postgresql://mluser:mlpassword@postgres-db:5432/mldb")
IMG_SIZE   = 150
CLASS_NAMES = ['Mountain', 'Glacier', 'Street', 'Sea', 'Forest', 'Buildings']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intel Image Classifier API",
    description="Classification d'images — CNN entraîné sur Spark Parquet",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------
# CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ---------------------------------------------------------------
models_cache: dict = {}
model_meta: dict   = {}

@app.on_event("startup")
async def load_models():
    global model_meta
    meta_path = Path(MODEL_DIR) / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            model_meta = json.load(f)
        logger.info(f"Méta-données chargées : {model_meta}")

    for fname in ["intel_model_cnn.h5", "intel_model_mlp.h5", "intel_model_bigcnn.h5"]:
        fpath = Path(MODEL_DIR) / fname
        if fpath.exists():
            try:
                models_cache[fname] = tf.keras.models.load_model(str(fpath))
                logger.info(f"Modèle chargé : {fname}")
            except Exception as e:
                logger.error(f"Erreur chargement {fname} : {e}")

def get_model(model_name: str = "intel_model_cnn.h5"):
    m = models_cache.get(model_name)
    if m is None:
        raise HTTPException(status_code=503, detail=f"Modèle '{model_name}' non disponible")
    return m

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def save_to_db(prediction: int, confidence: float, model_name: str):
    try:
        conn = psycopg2.connect(DB_URL)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (features, prediction, confidence, created_at) VALUES (%s, %s, %s, %s)",
            (json.dumps({"model": model_name}), prediction, confidence, datetime.utcnow())
        )
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        logger.warning(f"DB save failed : {e}")

# ---------------------------------------------------------------
# SCHÉMAS
# ---------------------------------------------------------------
class FeatureRequest(BaseModel):
    """Prédiction à partir d'un vecteur features (envoyé par Streamlit)"""
    features:   list[float]
    model_name: Optional[str] = "intel_model_cnn.h5"

class PredictionResponse(BaseModel):
    prediction:    int
    label:         str
    confidence:    float
    probabilities: list[float]

# ---------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(models_cache.keys()),
        "model_meta":    model_meta,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_from_features(request: FeatureRequest):
    """Prédiction à partir d'un vecteur features flatten (depuis Streamlit)."""
    model = get_model(request.model_name)

    expected = IMG_SIZE * IMG_SIZE * 3
    if len(request.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Attendu {expected} features (150×150×3), reçu {len(request.features)}"
        )

    X     = np.array(request.features, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3)
    probs = tf.nn.softmax(model.predict(X, verbose=0)[0]).numpy()
    idx   = int(np.argmax(probs))
    conf  = float(np.max(probs))

    save_to_db(idx, conf, request.model_name)

    return PredictionResponse(
        prediction=idx,
        label=CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx),
        confidence=conf,
        probabilities=[float(p) for p in probs],
    )

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_from_image(
    file:       UploadFile = File(...),
    model_name: str        = "intel_model_cnn.h5"
):
    """Prédiction directe à partir d'un fichier image uploadé."""
    model   = get_model(model_name)
    content = await file.read()
    image   = Image.open(io.BytesIO(content))
    X       = preprocess_image(image)

    probs = tf.nn.softmax(model.predict(X, verbose=0)[0]).numpy()
    idx   = int(np.argmax(probs))
    conf  = float(np.max(probs))

    save_to_db(idx, conf, model_name)

    return PredictionResponse(
        prediction=idx,
        label=CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx),
        confidence=conf,
        probabilities=[float(p) for p in probs],
    )

@app.get("/predictions/history")
def get_history(limit: int = 50):
    try:
        conn = psycopg2.connect(DB_URL)
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, prediction, confidence, created_at FROM predictions ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        return [{"id": r[0], "prediction": r[1], "confidence": r[2], "created_at": str(r[3])} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
def model_info():
    return model_meta if model_meta else {"detail": "Méta-données introuvables"}
