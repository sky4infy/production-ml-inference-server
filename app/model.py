"""
Model layer. Kept separate from main.py on purpose: in later phases this is
exactly the file that gets swapped out for raw tokenizer+model calls (for
batching) and ONNX/quantized versions, without touching the API code at all.
"""

import torch
from transformers import pipeline

# By default, PyTorch uses ALL available CPU cores for a single inference
# call (intra-op parallelism). That's fine for one request at a time, but
# under concurrent load, many requests each trying to claim every core at
# once causes contention rather than real parallelism - especially on a
# laptop CPU with only a handful of cores. Capping this to a small number
# leaves room for multiple requests to actually run alongside each other
# instead of fighting over the same cores. Worth experimenting with this
# number (try 1, 2, 4) and comparing load test results each time.
torch.set_num_threads(2)

_classifier = None


def get_classifier():
    """
    Lazily loads the model once and reuses it across requests.
    Without this, every request would reload the model from scratch -
    that alone can be a >1000x latency difference.
    """
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    return _classifier


def predict(text: str) -> dict:
    classifier = get_classifier()
    result = classifier(text)[0]
    return {"label": result["label"], "score": round(float(result["score"]), 4)}


def predict_batch(texts: list[str]) -> list[dict]:
    """
    Same model, but takes a LIST of texts and runs them through in one
    forward pass. The Hugging Face pipeline already supports batched input
    natively - this is the function the batcher will call once it has
    collected several requests together.
    """
    classifier = get_classifier()
    results = classifier(texts)
    return [
        {"label": r["label"], "score": round(float(r["score"]), 4)}
        for r in results
    ]