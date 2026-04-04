# Intel Image Classification — Pipeline Docker

Pipeline ML conteneurisé : Spark (Scala) → Keras/TF → FastAPI + Streamlit.

```
images brutes (train/ test/)
        ↓
spark-preprocessing  →  Parquet (features 150×150×3 + label)
        ↓
keras-training       →  intel_model_mlp.h5
                     →  intel_model_cnn.h5       ← défaut
                     →  intel_model_bigcnn.h5
                     →  model_meta.json
        ↓                        ↓
streamlit-app               api-rest  ←→  postgres-db
(http://localhost:8501)  (http://localhost:8000)
```

**6 classes :** Mountain · Glacier · Street · Sea · Forest · Buildings

---

## Structure attendue des données

```
ml-pipeline/
└── data/
    └── raw/
        ├── train/
        │   ├── Mountain/   ← images .jpg/.png
        │   ├── Glacier/
        │   ├── Street/
        │   ├── Sea/
        │   ├── Forest/
        │   └── Buildings/
        └── test/
            ├── Mountain/
            └── ...
```

---

## Démarrage

```bash
# 1. Place tes données
cp -r /ton/dataset/train  ml-pipeline/data/raw/train
cp -r /ton/dataset/test   ml-pipeline/data/raw/test

# 2. Lance tout
cd ml-pipeline
docker-compose up --build

# Accès
# → Streamlit  : http://localhost:8501
# → API Swagger : http://localhost:8000/docs
```

---

## Endpoints API

| Méthode | URL | Description |
|---|---|---|
| GET | `/health` | État + modèles chargés |
| POST | `/predict` | Prédiction via vecteur features (Streamlit) |
| POST | `/predict/image` | Prédiction via fichier image uploadé |
| GET | `/predictions/history` | Historique PostgreSQL |
| GET | `/model/info` | Méta-données des modèles |

---

## Commandes utiles

```bash
# Logs d'un service
docker-compose logs -f keras-training

# Relancer uniquement le preprocessing
docker-compose run --rm spark-preprocessing

# Relancer uniquement l'entraînement
docker-compose run --rm keras-training

# Arrêt complet
docker-compose down
```

---

## Réseaux Docker

| Réseau | Services |
|---|---|
| `preprocessing_net` | spark-preprocessing ↔ keras-training |
| `training_net` | keras-training ↔ api-rest |
| `app_net` | api-rest ↔ streamlit-app ↔ postgres-db |
