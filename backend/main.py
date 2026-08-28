"""
main.py

"""

import tempfile
import os

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from model import AudioCNN
from preprocessing import audio_file_to_model_input

app = FastAPI()

# Allow the React app (running on a different local port) to call this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # default Vite React dev server address
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the model ONCE, when the server starts up ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AudioCNN()
model.load_state_dict(torch.load("audio_cnn_model.pth", map_location=device))
model = model.to(device)
model.eval()  # inference mode - same reasoning as your dev-set evaluation

LABELS = {0: "bonafide", 1: "spoof"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded audio file, runs it through the same
    preprocessing + model pipeline used in the notebook, and returns
    a prediction.
    """
    # UploadFile arrives as an in-memory stream. librosa.load() needs an
    # actual file path, so we write it to a temporary file on disk first.
    suffix = os.path.splitext(file.filename)[1]  # e.g. ".flac", ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        model_input = audio_file_to_model_input(tmp_path)

        if model_input is None:
            return {"error": "Could not process this audio file."}

        model_input = model_input.to(device)

        with torch.no_grad():
            output = model(model_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            "prediction": LABELS[predicted_class],
            "confidence": round(confidence, 4),
        }
    finally:
        os.remove(tmp_path)  # clean up the temporary file regardless of outcome


@app.get("/")
async def root():
    return {"status": "ASVspoof detector API is running"}
