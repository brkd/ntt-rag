# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh 

# Copy only dependency files first (best for caching)
COPY requirements.txt ./


# Install dependencies (cached unless requirements.txt changes)
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# This changes often, so copy last
COPY pyproject.toml ./
COPY src/ ./src/

# Install your package without reinstalling deps
RUN /root/.local/bin/uv pip install --system . --no-deps

# Expose FastAPI port
EXPOSE 9632

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "9632"]