# Worker RunPod Serverless – lettura conferme d'ordine (vedi serverless/README.md)
# Immagine vLLM ufficiale con CUDA 12.9: funziona su GPU da Turing a Blackwell (driver >= 535).
FROM vllm/vllm-openai:v0.32.0-x86_64-cu129

WORKDIR /app

COPY serverless/requirements.txt /app/serverless/requirements.txt
RUN uv pip install --system --no-cache -r /app/serverless/requirements.txt

# solo il codice necessario: il dataset e il modello NON vanno nell'immagine
COPY tools/build_dataset.py tools/infer.py tools/controlli.py /app/tools/
COPY serverless/handler.py /app/serverless/handler.py

# il modello si carica dal network volume (montato dai worker serverless in /runpod-volume)
# la cache di compilazione di vLLM sta sul volume: solo il primo worker (per tipo di GPU) compila
ENV MODEL_PATH=/runpod-volume/orderRecognition/runs/qwen3-4b-r32/merged \
    VLLM_CACHE_ROOT=/runpod-volume/vllm-cache \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# l'immagine base parte con "vllm serve": la sostituiamo con l'handler RunPod
ENTRYPOINT []
CMD ["python3", "-u", "/app/serverless/handler.py"]
