"""
main.py
~~~~~~~

Entry point for the FastAPI app. This is what uvicorn imports and runs.

Right now this file does the absolute minimum: create an "app" object
and give it one route, so we can confirm the server boots and responds
before wiring in the actual neural network.
"""

from fastapi import FastAPI

# This is the ASGI application object. uvicorn looks for a variable
# named `app` in this file when we run `uvicorn main:app`.
app = FastAPI()


# @app.get("/") registers this function as the handler for
# HTTP GET requests to the root path "/".
# FastAPI calls this function whenever a request matches, and
# whatever it returns gets converted to JSON automatically.
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Digit classifier API is running"}
