"""
schemas.py
~~~~~~~~~~

Pydantic models describing the shape of API requests/responses.
Kept separate from app.py so routing logic doesn't get tangled up
with data validation rules - if the request/response shape changes,
this is the only file that should need editing.
"""

from pydantic import BaseModel, Field


class DigitInput(BaseModel):
    """Request body for POST /predict.

    Exactly 784 floats: one per pixel in a flattened 28x28 MNIST
    image, each normalized to the 0.0-1.0 range (same preprocessing
    digit_classifier.data.loader already applies to training data).
    """
    pixels: list[float] = Field(min_length=784, max_length=784)


class PredictionOutput(BaseModel):
    """Response body for POST /predict."""
    predicted_digit: int
    confidence: float


class PredictionRecord(BaseModel):
    """One row from the predictions log, as returned by GET /predictions.
    Deliberately excludes the raw pixels - this is a summary view, not
    a way to pull training data back out through the API."""
    id: int
    timestamp: str
    predicted_digit: int
    confidence: float
