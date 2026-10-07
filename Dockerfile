FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Mount or bake a trained model at this path (models/*.joblib is git-ignored)
ENV VSD_MODEL_PATH=/models/model.joblib
EXPOSE 8000
RUN useradd -m app && chown -R app /app
USER app
CMD ["uvicorn", "vsd.api:app", "--host", "0.0.0.0", "--port", "8000"]
