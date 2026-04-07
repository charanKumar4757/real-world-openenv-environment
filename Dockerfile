# Use a stable, explicit version to avoid registry manifest errors
FROM python:3.10.13-slim-bookworm

# Set environment variables to ensure Python output is logged correctly
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install essential system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and copy requirements first
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of your project
COPY . /app

# Default port for Hugging Face Spaces
EXPOSE 7860

# Use the FastAPI server entry point
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
