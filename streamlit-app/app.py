"""
streamlit-app/app.py
Interface de classification d'images Intel.
Délègue 100 % des prédictions à api-rest — aucun modèle chargé en local.
"""

import os
import requests
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
API_URL     = os.getenv("API_URL", "http://api-rest:8000")
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
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Statut API")

    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        health = r.json()
        if "intel_model_cnn.h5" in health.get("models_loaded", []):
            st.success("✅ API connectée — CNN chargé")
        else:
            st.warning("⚠️ API connectée mais CNN non disponible")
    except Exception:
        st.error("❌ API injoignable")
        health = {}

    st.markdown("---")
    page = st.radio("Navigation", ["🔮 Prédiction", "📈 Historique", "ℹ️ À propos"])

# ---------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0

def predict_api(image: Image.Image) -> dict:
    arr      = preprocess(image)
    features = arr.flatten().tolist()
    r = requests.post(
        f"{API_URL}/predict",
        json={"features": features, "model_name": "intel_model_cnn.h5"},
        timeout=15
    )
    r.raise_for_status()
    return r.json()

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
            with st.spinner("Classification en cours..."):
                try:
                    result     = predict_api(image)
                    label_idx  = result["prediction"]
                    confidence = result["confidence"]
                    probs      = result["probabilities"]

                    st.subheader(f"Résultat : **{CLASS_NAMES[label_idx]}**")
                    st.metric("Confiance", f"{confidence * 100:.2f}%")
                    st.markdown("**Distribution des probabilités :**")
                    chart_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
                    st.bar_chart(chart_data)

                except Exception as e:
                    st.error(f"Erreur API : {e}")
    else:
        st.info("Uploadez une image pour commencer la classification.")

# ---------------------------------------------------------------
# PAGE : HISTORIQUE
# ---------------------------------------------------------------
elif page == "📈 Historique":
    st.subheader("Historique des prédictions")
    try:
        r = requests.get(f"{API_URL}/predictions/history?limit=50", timeout=5)
        r.raise_for_status()
        history = r.json()
        if history:
            import pandas as pd
            import plotly.express as px
            df = pd.DataFrame(history)
            df["classe"] = df["prediction"].apply(
                lambda x: CLASS_NAMES[x] if x < len(CLASS_NAMES) else str(x)
            )
            st.dataframe(df[["id", "classe", "confidence", "created_at"]], use_container_width=True)
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
    | **keras-training** | TensorFlow 2.15 | CNN — classification 6 classes |
    | **api-rest** | FastAPI | Endpoint `/predict` + historique PostgreSQL |
    | **streamlit-app** | Streamlit | Interface web upload & prédiction |
    | **postgres-db** | PostgreSQL 15 | Stockage des prédictions |

    **Classes détectées :** {", ".join(CLASS_NAMES)}

    **Taille d'entrée :** {IMG_SIZE}×{IMG_SIZE}×3 pixels
    """)