
# -----------------------------
# Dockerfile for Mortar CS Predictor (Continuous GO Model)
# -----------------------------
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app_streamlit_realdata.py", "--server.port=8501", "--server.address=0.0.0.0"]
