# Intel Image Classification — Pipeline Docker

Pipeline ML conteneurisé : Spark (Scala) → Keras/TF → FastAPI + Streamlit.

```
images brutes (train/ test/)
        ↓
spark-preprocessing  →  Parquet (features 150×150×3 + label)
        ↓
keras-training       →  intel_model_cnn.h5       ← défaut
        ↓                        ↓
streamlit-app               api-rest  ←→  postgres-db
(http://localhost:8501)  (http://localhost:8000)
```

**6 classes :** Mountain · Glacier · Street · Sea · Forest · Buildings

---

## Structure attendue des données

```
docker-scala-project/
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
# 1. Placer les données
cp -r /le/dataset/train  docker-scala-project/data/raw/train
cp -r /le/dataset/test   docker-scala-project/data/raw/test

# 2. Lancer tout
cd docker-scala-project
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
| `app_net` | api-rest ↔ streamlit-app ↔ postgres-db |
