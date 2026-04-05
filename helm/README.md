# docker-scala-project — Helm Chart

## Installation

```bash
# 1. Builder les images localement
docker build -t spark-preprocessing:latest ./spark-preprocessing
docker build -t training:latest      ./keras-training
docker build -t api-rest:latest            ./api-rest
docker build -t streamlit-app:latest       ./streamlit-app

# 2. Créer le namespace
kubectl create namespace docker-scala-project

# 3. Installer le chart
helm install docker-scala-project ./helm

# 4. Suivre le démarrage
kubectl get pods -n docker-scala-project -w
```

## Accès
- Streamlit : http://localhost:30501
- API REST  : http://localhost:30800
- API docs  : http://localhost:30800/docs

## Désinstaller
```bash
helm uninstall docker-scala-project
kubectl delete namespace docker-scala-project
```