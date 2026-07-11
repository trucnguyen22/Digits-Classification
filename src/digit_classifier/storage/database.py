"""
database.py
~~~~~~~~~~~

SQLite-backed logging of prediction requests.

This is deliberately separate from digit_classifier.data, which loads
the *training* data (MNIST). This module records what happens at
*inference* time: every digit a real person submits through /predict,
what the network predicted, and when. Over time that becomes a
growing dataset of genuine user-drawn digits - meaningfully different
from MNIST's scanned handwriting samples - which could later be used
to check for model drift or to fine-tune the network on real input.

Uses the standard library's sqlite3 module directly. The schema is
one table with one write path and one read path, which doesn't need
an ORM's complexity.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# This file lives at: <repo_root>/src/digit_classifier/storage/database.py
# Same repo-root-relative pattern as data.loader's DEFAULT_DATA_PATH and
# api.app's WEIGHTS_PATH - don't depend on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = _REPO_ROOT / "db" / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predicted_digit INTEGER NOT NULL,
    confidence REAL NOT NULL,
    pixels TEXT NOT NULL
);
"""


@contextmanager
def _connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path=DEFAULT_DB_PATH):
    """Create the predictions table if it doesn't exist yet. Safe to
    call on every app startup - CREATE TABLE IF NOT EXISTS is a no-op
    once the table's already there."""
    with _connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def log_prediction(pixels, predicted_digit, confidence, db_path=DEFAULT_DB_PATH):
    """Record one prediction. `pixels` is stored as JSON text rather
    than a binary blob - larger on disk, but human-inspectable and
    trivial to reload with json.loads() later, which matters more for
    a dataset you intend to actually look at and reuse."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, predicted_digit, confidence, pixels) "
            "VALUES (?, ?, ?, ?)",
            (timestamp, predicted_digit, confidence, json.dumps(pixels)),
        )
        conn.commit()


def get_recent_predictions(limit=20, db_path=DEFAULT_DB_PATH):
    """Return the most recent predictions, newest first. Excludes the
    pixels column by default - 784 floats per row is unwieldy for a
    quick summary view; this is for 'what's been happening', not for
    pulling training data back out."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, timestamp, predicted_digit, confidence "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
