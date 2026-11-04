
FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Dossier de travail
WORKDIR /app

# Installer les dépendances
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement les fichiers nécessaires
COPY app_api.py .
COPY models/ ./models/

# Port utilisé par l'API
EXPOSE 7860

# Lancer FastAPI
CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port ${PORT}"]
