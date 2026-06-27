"""
ONNX Runtime inference layer.

Identical public interface to model.py (predict, predict_batch, get_classifier)
so main.py can swap between backends with minimal changes.

The key difference from model.py:
- Tokenization still uses HuggingFace (fast, proven, no reason to replace)
- The model forward pass is replaced with an onnxruntime InferenceSession
- Session is loaded once at startup, same lazy-load pattern as model.py
"""

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as ort

MODEL_NAME  = "distilbert-base-uncased-finetuned-sst-2-english"
INT8_PATH   = Path("models") / "model_int8.onnx"
MAX_SEQ_LEN = 128

# Label mapping for DistilBERT SST-2
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

_tokenizer = None
_session   = None


def _load():
    global _tokenizer, _session
    if _session is not None:
        return

    if not INT8_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {INT8_PATH}. "
            "Run `python export_onnx.py` first."
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    session_opts = ort.SessionOptions()
    # Cap threads to avoid contention with concurrent async requests.
    # Same reasoning as torch.set_num_threads(2) in model.py.
    session_opts.intra_op_num_threads = 2
    session_opts.inter_op_num_threads = 1
    # Graph optimization: apply all available ORT optimizations to the
    # session graph at load time (constant folding, node fusion, etc.)
    session_opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    _session = ort.InferenceSession(
        str(INT8_PATH),
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )


def get_classifier():
    """Matches the model.py interface used in main.py startup."""
    _load()
    return _session


def _softmax(logits: np.ndarray) -> np.ndarray:
    # Numerically stable softmax: subtract max before exp to prevent overflow
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _tokenize(texts: list[str]) -> dict:
    return _tokenizer(
        texts,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        truncation=True,
    )


def predict(text: str) -> dict:
    _load()
    inputs  = _tokenize([text])
    outputs = _session.run(
        ["logits"],
        {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )
    probs     = _softmax(outputs[0])[0]   # shape: (2,)
    label_idx = int(np.argmax(probs))
    return {
        "label": ID2LABEL[label_idx],
        "score": round(float(probs[label_idx]), 4),
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """
    Batch inference: same model call, multiple inputs.
    This is what the batcher calls - the whole point of batching is that
    this costs barely more than predict() for a single input.
    """
    _load()
    inputs  = _tokenize(texts)
    outputs = _session.run(
        ["logits"],
        {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )
    probs_batch = _softmax(outputs[0])   # shape: (batch_size, 2)
    results = []
    for probs in probs_batch:
        label_idx = int(np.argmax(probs))
        results.append({
            "label": ID2LABEL[label_idx],
            "score": round(float(probs[label_idx]), 4),
        })
    return results