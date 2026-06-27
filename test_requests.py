"""
Run this AFTER the server is up (uvicorn app.main:app --reload).
Sends a handful of requests one at a time and prints latency for each,
plus the average. Save this output - it's your "before batching" baseline
number that you'll compare against once batching is added in phase 2.
"""

import time
import requests

URL = "http://127.0.0.1:8000/predict"

SAMPLE_TEXTS = [
    "I absolutely loved this movie, best one this year.",
    "This was a complete waste of my time.",
    "The food was okay, nothing special.",
    "Amazing service, will definitely come back!",
    "I'm so disappointed with this product.",
    "Pretty good overall, would recommend.",
    "Terrible experience, never again.",
    "It exceeded all my expectations.",
    "Mediocre at best, not impressed.",
    "One of the best purchases I've made this year.",
]

if __name__ == "__main__":
    latencies = []
    for text in SAMPLE_TEXTS:
        start = time.perf_counter()
        response = requests.post(URL, json={"text": text})
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        data = response.json()
        print(f"{elapsed_ms:6.1f}ms | {data['label']:8s} ({data['score']:.3f}) | {text[:50]}")

    print(f"\nAverage round-trip latency: {sum(latencies) / len(latencies):.1f}ms")
    print(f"Min: {min(latencies):.1f}ms  Max: {max(latencies):.1f}ms")