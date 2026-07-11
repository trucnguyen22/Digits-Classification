# Digit Classifier

A handwritten digit classifier built from scratch: the neural network itself
is implemented with nothing but NumPy (no PyTorch, no TensorFlow, no
scikit-learn), trained on MNIST, and served through a FastAPI backend with a
browser-based drawing frontend.

## Why this project exists

Most ML tooling hides the actual mechanics of a neural network behind a
`.fit()` call. This project goes the other way: forward propagation,
backpropagation, and gradient descent are all implemented directly, so the
math (the chain rule derivation, the per-layer error signal, the weight
update rule) is fully visible and understood rather than delegated to a
framework.

The second half of the project is the part frameworks usually skip past
entirely: turning a trained model into something that actually runs as a
service. That means an HTTP API with request validation, a real frontend
that captures and preprocesses user input the same way the training data
was preprocessed, a persisted/reloadable model artifact instead of retraining
on every request, an automated correctness check (numerical gradient
checking) instead of just trusting that training loss went down, a pytest
regression suite, prediction logging, and a cloud deployment - the full path
from "the math works" to "a person can use this in a browser."

The implementation follows the structure introduced in Michael Nielsen's
[*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/),
refactored into an object-oriented design and extended with the API, testing,
logging, and deployment work described below.

## What it does

Draw a digit (0-9) in the browser, and the network predicts what you drew,
along with its confidence. Under the hood: your 280x280 canvas drawing is
downsampled to 28x28 and normalized to match MNIST's format, sent to the
API, run through the trained network, and the prediction is sent back.

## How the network works

Four classes, each with a single responsibility:

- **`Layer`** - a fully-connected layer. Owns its weights and biases, knows
  how to feed data forward through itself, and how to compute its own
  gradients given an error signal.
- **`NeuralNetwork`** - a stack of `Layer`s. Knows the architecture and how
  to run feedforward/backpropagation across all of them, but nothing about
  how gradients get applied to weights.
- **`Optimizer`** - owns the parameter update rule (currently vanilla SGD).
  Swapping in momentum or Adam later means writing a new `Optimizer`
  subclass; nothing else changes.
- **`Trainer`** - owns the training loop: epochs, mini-batching, shuffling,
  and progress reporting.

## Verifying correctness

Analytic backpropagation is easy to get subtly wrong (a transpose in the
wrong place, a sign error) in a way that still trains a network, just worse
than it should. Two things exist specifically to catch that:

- **Numerical gradient checking** (`src/digit_classifier/diagnostics/gradient_checker.py`) -
  estimates each gradient independently from the definition of a derivative
  (a centered finite difference), with no dependency on the backprop code
  being checked, and compares it to the analytic gradient via relative
  error. Methodology, design tradeoffs, and actual results are documented in
  [`DEBUGGING.md`](./DEBUGGING.md).
- **A pytest suite** (`tests/`) covering gradient correctness, forward/backward
  pass shapes across multiple architectures, and a training convergence
  sanity check (loss should trend downward over epochs on a small synthetic
  dataset).

## Prediction logging

Every call to `/predict` is logged to a SQLite database
(`src/digit_classifier/storage/database.py`): the submitted pixels, the
predicted digit, the confidence, and a timestamp. This is separate from
`digit_classifier.data`, which loads MNIST's *training* data - this module
records real *inference-time* input instead.

Beyond basic observability, this builds a growing dataset of genuine
user-drawn digits - meaningfully different from MNIST's scanned handwriting
samples - which could be used later to check for model drift or to fine-tune
the network on real-world input. Recent predictions (without the raw pixels)
can be viewed via `GET /predictions`.

## Project structure

```
.
├── data/
│   └── mnist.pkl.gz                  # training data
├── models/
│   └── model_weights.pkl             # trained weights (generated, gitignored)
├── db/
│   └── predictions.db                # prediction log (generated, gitignored)
├── src/digit_classifier/
│   ├── model/network.py              # Layer, NeuralNetwork, Optimizer, Trainer
│   ├── data/loader.py                # MNIST loading/preprocessing
│   ├── diagnostics/gradient_checker.py
│   ├── storage/database.py           # SQLite prediction logging
│   ├── api/
│   │   ├── app.py                    # FastAPI routes
│   │   ├── schemas.py                # request/response models
│   │   └── static/index.html         # drawing frontend
│   └── scripts/train.py              # one-off training script
├── tests/                            # pytest suite
├── DEBUGGING.md                      # gradient-check methodology & results
└── pyproject.toml
```

## Running it locally

Requires Python 3.10+.

```bash
git clone https://github.com/trucnguyen22/Digits-Classification.git
cd Digits-Classification
python3 -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -e .
```

Train the network once (produces `models/model_weights.pkl` - the API loads
this at startup rather than retraining on every request):

```bash
python -m digit_classifier.scripts.train
```

Start the API:

```bash
uvicorn digit_classifier.api.app:app --reload
```

Visit `http://127.0.0.1:8000/` to draw a digit and get a prediction.
`http://127.0.0.1:8000/docs` gives an interactive API reference (auto-generated
by FastAPI), and `http://127.0.0.1:8000/health` is a plain health check.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -v
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the drawing frontend |
| `/health` | GET | Health check |
| `/predict` | POST | Body: `{"pixels": [784 floats, 0.0-1.0]}`. Returns `{"predicted_digit": int, "confidence": float}` |
| `/predictions` | GET | Recent logged predictions (`?limit=N`, default 20). Returns id, timestamp, predicted digit, and confidence for each - no raw pixels |

## Deployment

The API is designed to run unmodified on a cloud VM: same `uvicorn` command,
bound to `0.0.0.0` instead of `127.0.0.1` so it's reachable externally. It's
currently deployed on an AWS EC2 instance (Ubuntu, t3.micro).

## Roadmap

- **GitHub Issues / PR workflow** - track ongoing work through issues and
  pull requests rather than direct commits, as a more realistic software
  development workflow.
