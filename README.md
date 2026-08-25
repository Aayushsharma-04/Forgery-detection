# Image Forgery Detection

Detects whether an image has been digitally tampered (spliced/copy-moved) using Error Level Analysis (ELA) preprocessing and a fine-tuned ResNet18.

## Approach
- **Preprocessing:** Error Level Analysis — recompresses each image and diffs it against the original; regions with inconsistent compression history (likely tampering) appear as bright patches.
- **Model:** ResNet18, transfer learning in two phases:
  - Phase 1: frozen backbone, trained only the final layer — 86.7% val accuracy, 88.3% test accuracy.
  - Phase 2: fine-tuned layer3/layer4 with differential learning rates (1e-5 backbone, 1e-4 head) — improved to 90.4% test accuracy.
- **Dataset:** CASIA v2 (7,491 authentic, 5,123 tampered), stratified 70/15/15 split.

## Results
- Test accuracy: 90.4%, F1 (macro avg): 0.90
- Precision/Recall — authentic: 0.94/0.89, tampered: 0.85/0.92
- Grad-CAM visualizations confirm the model attends to spatially coherent tampered regions, not background artifacts.

## Run locally

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000


## Run with Docker

docker build -t forgery-detection .
docker run -p 8000:8000 forgery-detection


## API
- `GET /health` — service status
- `POST /predict` — upload an image, returns `{label, confidence}`

## Limitations
- ELA effectiveness can vary with source image quality/compression history.
- Trained on general photography (CASIA v2), not document-specific tampering.