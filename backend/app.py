"""
Steel Defect Inspector — inference server.

Loads your best.pt YOLO model once at startup and exposes a small API
for the frontend to call. Also serves the frontend itself, so the whole
app runs from a single `uvicorn app:app` command.

Setup:
    1. Drop your trained weights in this folder as best.pt
    2. pip install -r requirements.txt
    3. uvicorn app:app --reload --port 8000
    4. Open http://localhost:8000
"""

import io
import time
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path("C:/Ramya/Rvce/4thSemProjects/MaterialScience/best-after40.pt")
FRONTEND_DIR = Path("C:/Ramya/Rvce/4thSemProjects/MaterialScience/steel-defect-inspector/frontend")

# Boxes below this confidence are dropped before they ever reach the
# frontend. Kept low on purpose — the frontend has its own threshold
# slider so people can tighten/loosen the cutoff without a new request.
MIN_CONFIDENCE = 0.05

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No model found at {MODEL_PATH}.\n"
        f"Double-check that path and restart the server."
    )

if not FRONTEND_DIR.exists():
    print(f"WARNING: frontend folder not found at {FRONTEND_DIR} — "
          f"the UI won't be served, only the API.")

model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names  # e.g. {0: 'scratch', 1: 'pitted_surface', ...}

app = FastAPI(title="Steel Defect Inspector")

# Wide-open CORS is fine for local/dev use. Tighten allow_origins if you
# deploy this somewhere other people can reach.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "classes": list(CLASS_NAMES.values()),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't read that file as an image.")

    started = time.time()
    results = model.predict(image, conf=MIN_CONFIDENCE, verbose=False)
    inference_ms = round((time.time() - started) * 1000)

    result = results[0]
    detections = []
    counts = Counter()

    for box in result.boxes:
        cls_id = int(box.cls.item())
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        detections.append(
            {
                "class_name": cls_name,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )
        counts[cls_name] += 1

    return {
        "image_width": image.width,
        "image_height": image.height,
        "inference_ms": inference_ms,
        "detections": detections,
        "summary": dict(counts),
        "total_defects": len(detections),
    }


# Serve the frontend at "/" so the API and UI live behind one port.
# Routes declared above still take priority over this catch-all mount.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
