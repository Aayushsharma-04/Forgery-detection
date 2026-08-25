# app/main.py
import io
import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from contextlib import asynccontextmanager

from data.transforms import val_test_transforms
from data.ela import compute_ela
from app.schemas import HealthResponse, PredictionResponse

MODEL_PATH = "model/artifacts/best_model_finetuned.pth"
LABEL_MAP = {0: "authentic", 1: "tampered"}

state = {"model": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    state["model"] = model
    state["device"] = device
    yield
    state["model"] = None


app = FastAPI(
    title="Image Forgery Detection API",
    description="Detects whether an uploaded image is authentic or tampered using ELA + fine-tuned ResNet18.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=state["model"] is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    ela_image = compute_ela(pil_image)
    input_tensor = val_test_transforms(ela_image).unsqueeze(0).to(state["device"])

    with torch.no_grad():
        logits = state["model"](input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_idx].item())

    return PredictionResponse(label=LABEL_MAP[predicted_idx], confidence=round(confidence, 4))