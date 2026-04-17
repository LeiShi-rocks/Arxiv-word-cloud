FROM python:3.11-slim

# System libs needed by wordcloud (font rendering) and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so it's baked into the image
# (avoids a network call on every cold start)
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True)"

# Copy application code
COPY app.py .
COPY static/ ./static/

# HF Spaces routes external traffic to port 7860
EXPOSE 7860

ENV PORT=7860
ENV ENV=production

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
