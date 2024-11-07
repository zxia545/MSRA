python -m vllm.entrypoints.openai.api_server \
    --model=${MODEL_LOCAL_PATH_OR_HUGGINGFACE_NAME} \
    --served-model-name=gpt-4 \
    --tensor-parallel-size=4 \
    --port=8003 &
