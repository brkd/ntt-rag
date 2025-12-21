# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files AND source code (needed for editable install)
COPY pyproject.toml ./
COPY src/ ./src/

# Install uv and dependencies in one go, then clean up
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --system -e . && \
    rm -rf /root/.cargo

# Expose FastAPI port
EXPOSE 9632

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "9632"]