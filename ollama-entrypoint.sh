#!/bin/bash
set -e

# Start Ollama in the background
/bin/ollama serve &


# Pull models if specified
if [ -n "$NTT_RAG_LLM_MODEL" ]; then
    IFS=',' read -ra MODELS <<< "$NTT_RAG_LLM_MODEL"
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs) # trim whitespace
        echo "Pulling model: $model"
        ollama pull "$model"
    done
    echo "All models pulled successfully!"
else
    echo "No models specified in NTT_RAG_LLM_MODEL environment variable"
fi

wait