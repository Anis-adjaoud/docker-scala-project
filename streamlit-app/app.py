"""
streamlit-app/app.py
Interface de classification d'images Intel — charge le modèle CNN Keras
et communique avec l'API REST pour les prédictions.
"""

import os
import json
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
MODEL_DIR  = os.getenv("MODEL_PATH", "/data/models")
API_URL    = os.getenv("API_URL",    "http://api-rest:8000")
META_PATH  = str(Path(MODEL_DIR) / "model_meta.json")

IMG_SIZE    = 150
CLASS_NAMES = ['Mountain', 'Glacier', 'Street', 'Sea', 'Forest', 'Buildings']

# ---------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Intel Image Classifier",
    page_icon="🖼️",
    layout="wide"
)
st.title("🖼️ Classification d'Images avec Spark & Deep Learning")

# ---------------------------------------------------------------
# CHARGEMENT MODÈLE LOCAL
# ---------------------------------------------------------------
@st.cache_resource
def load_model(model_name: str = "intel_model_cnn.h5"):
    path = str(Path(MODEL_DIR) / model_name)
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

@st.cache_data(ttl=60)
def load_meta() -> dict:
    try:
        with open(META_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

meta = load_meta()

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")

    available_models = {
        "CNN (recommandé)": "intel_model_cnn.h5",
        "BigCNN":           "intel_model_bigcnn.h5",
        "MLP":              "intel_model_mlp.h5",
    }
    selected_label = st.selectbox("Modèle", list(available_models.keys()))
    model_file     = available_models[selected_label]
    model          = load_model(model_file)

    st.markdown("---")
    st.header("📊 Performances")

    if meta and "models" in meta:
        for mname, minfo in meta["models"].items():
            st.metric(
                label=mname,
                value=f"{minfo.get('test_accuracy', 0):.2%}",
                help=f"Epochs: {minfo.get('epochs_run')} | Loss: {minfo.get('test_loss', 0):.4f}"
            )
    
    if model:
        st.success(f"✅ {model_file} chargé")
    else:
        st.warning("⚠️ Modèle introuvable — lance d'abord keras-training.")

    st.markdown("---")
    page = st.radio("Navigation", ["🔮 Prédiction", "📈 Historique", "ℹ️ À propos"])

# ---------------------------------------------------------------
# HELPER : prétraitement identique à Spark (resize 150x150, /255)
# ---------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_local(image: Image.Image):
    arr  = preprocess(image)
    pred = model.predict(arr, verbose=0)
    score = tf.nn.softmax(pred[0]).numpy()
    return int(np.argmax(score)), float(np.max(score)), score.tolist()

def predict_api(image: Image.Image):
    arr     = preprocess(image)
    features = arr.flatten().tolist()
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"API indisponible ({e}) — prédiction locale utilisée.")
        return None

# ---------------------------------------------------------------
# PAGE : PRÉDICTION
# ---------------------------------------------------------------
if page == "🔮 Prédiction":
    st.subheader("Choisissez une image à classifier")

    uploaded = st.file_uploader(
        "Glissez une image (jpg, jpeg, png)...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Image uploadée", use_column_width=True)

        with col2:
            if not model:
                st.error("Aucun modèle disponible. Lance d'abord l'entraînement.")
            else:
                with st.spinner("Classification en cours..."):
                    # Tentative API, fallback local
                    api_result = predict_api(image)
                    if api_result:
                        label_idx  = api_result["prediction"]
                        confidence = api_result["confidence"]
                        probs      = api_result["probabilities"]
                    else:
                        label_idx, confidence, probs = predict_local(image)

                st.subheader(f"Résultat : **{CLASS_NAMES[label_idx]}**")
                st.metric("Confiance", f"{confidence * 100:.2f}%")

                st.markdown("**Distribution des probabilités :**")
                chart_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
                st.bar_chart(chart_data)

    else:
        st.info("Uploadez une image pour commencer la classification.")

# ---------------------------------------------------------------
# PAGE : HISTORIQUE
# ---------------------------------------------------------------
elif page == "📈 Historique":
    st.subheader("Historique des prédictions (API)")
    try:
        r = requests.get(f"{API_URL}/predictions/history?limit=50", timeout=5)
        r.raise_for_status()
        history = r.json()
        if history:
            import pandas as pd
            df = pd.DataFrame(history)
            df["classe"] = df["prediction"].apply(
                lambda x: CLASS_NAMES[x] if x < len(CLASS_NAMES) else str(x)
            )
            st.dataframe(df[["id", "classe", "confidence", "created_at"]], use_container_width=True)

            import plotly.express as px
            fig = px.histogram(df, x="classe", title="Distribution des prédictions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune prédiction enregistrée pour le moment.")
    except Exception as e:
        st.warning(f"Impossible de contacter l'API : {e}")

# ---------------------------------------------------------------
# PAGE : À PROPOS
# ---------------------------------------------------------------
elif page == "ℹ️ À propos":
    st.subheader("Architecture du pipeline")
    st.markdown(f"""
    | Composant | Technologie | Rôle |
    |---|---|---|
    | **spark-preprocessing** | Spark 3.5 + Scala | Resize 150×150, normalisation, Parquet |
    | **keras-training** | TensorFlow 2.15 | MLP / CNN / BigCNN — classification 6 classes |
    | **api-rest** | FastAPI | Endpoint `/predict` + historique PostgreSQL |
    | **streamlit-app** | Streamlit | Interface web upload & prédiction |
    | **postgres-db** | PostgreSQL 15 | Stockage des prédictions |

    **Classes détectées :** {", ".join(CLASS_NAMES)}

    **Taille d'entrée :** {IMG_SIZE}×{IMG_SIZE}×3 pixels
    """)

    if meta:
        st.json(meta)
