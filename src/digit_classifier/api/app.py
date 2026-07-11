"""
app.py
~~~~~~

The FastAPI application: routing and request/response wiring only.
Validation shapes live in schemas.py, neural-network logic lives in
digit_classifier.model.network, and prediction logging lives in
digit_classifier.storage.database - this file's only job is to
connect HTTP requests to those pieces and hand back a response.
"""

import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from digit_classifier.model.network import NeuralNetwork
from digit_classifier.api.schemas import DigitInput, PredictionOutput, PredictionRecord
from digit_classifier.storage.database import init_db, log_prediction, get_recent_predictions

logger = logging.getLogger(__name__)

# This file lives at: <repo_root>/src/digit_classifier/api/app.py
# so the repo root is three directories up. Same pattern as
# digit_classifier.data.loader's DEFAULT_DATA_PATH, for the same
# reason: don't depend on whatever directory uvicorn happens to be
# launched from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_PATH = _REPO_ROOT / "models" / "model_weights.pkl"

# The frontend is a single self-contained HTML file (canvas + JS)
# living next to this module, not a separate app/server. Serving it
# from the same FastAPI process means one port, one process to deploy,
# and no CORS configuration needed since the page and the API it calls
# share the same origin.
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI()

# Load the trained network ONCE, when the server process starts - not
# on every request. Training is slow (minutes); inference is fast
# (milliseconds). If model_weights.pkl doesn't exist yet, the app still
# starts (so `/` works), but `network` stays None and /predict will
# return a clear error instead of crashing at import time.
try:
    network = NeuralNetwork.load(WEIGHTS_PATH)
except FileNotFoundError:
    network = None

# Create the predictions table if it doesn't exist yet. Safe to run on
# every startup - a no-op once the table's already there.
init_db()


@app.get("/")
def serve_frontend():
    """The drawing UI - what a person visiting the site actually sees."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Digit classifier API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input: DigitInput):
    if network is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model found at {WEIGHTS_PATH}. Run scripts/train.py first.",
        )

    # network.feedforward() expects a (784, 1) numpy column vector -
    # the same shape digit_classifier.data.loader produces for
    # training. The incoming pixels list is flat (length 784), so we
    # reshape it.
    x = np.array(input.pixels).reshape(784, 1)

    output_activations = network.feedforward(x)  # shape (10, 1)
    predicted_digit = int(np.argmax(output_activations))
    confidence = float(output_activations[predicted_digit, 0])

    # Logging is secondary to the actual prediction - if the database
    # write fails for any reason, the user should still get their
    # prediction rather than a 500 error over a logging problem.
    try:
        log_prediction(
            pixels=input.pixels,
            predicted_digit=predicted_digit,
            confidence=confidence,
        )
    except Exception:
        logger.exception("Failed to log prediction to database")

    return PredictionOutput(predicted_digit=predicted_digit, confidence=confidence)


@app.get("/predictions", response_model=list[PredictionRecord])
def recent_predictions(limit: int = 20):
    return get_recent_predictions(limit=limit)
