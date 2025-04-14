FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /
COPY app/ ./app/
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Pre-download the model so it can be built into the Docker image.
RUN pip install --no-cache-dir sentence-transformers==4.0.2 hf_xet huggingface_hub[hf_xet]
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "./app/naics_classifier.py"]