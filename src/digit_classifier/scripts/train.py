"""
train.py
~~~~~~~~

Run this ONCE (or whenever you want to retrain) to train the network
and persist its weights to disk. The FastAPI app never trains - it
just loads the file this script produces.

Usage (from the repo root, with the package installed - see README):
    python -m digit_classifier.scripts.train
"""

from pathlib import Path

from digit_classifier.data.loader import load_data_wrapper
from digit_classifier.model.network import NeuralNetwork, Optimizer, Trainer

_REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_PATH = _REPO_ROOT / "models" / "model_weights.pkl"


def main():
    training_data, validation_data, test_data = load_data_wrapper()

    network = NeuralNetwork([784, 30, 10])
    optimizer = Optimizer(learning_rate=3.0)
    trainer = Trainer(network, optimizer)

    trainer.train(training_data, epochs=30, mini_batch_size=10, test_data=test_data)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    network.save(WEIGHTS_PATH)
    print(f"Saved trained weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
