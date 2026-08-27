# 🔩 Steel Defect Inspector

A YOLO-powered web app for detecting surface defects on steel — drop an
image in, get labeled bounding boxes and a defect breakdown back in
milliseconds. FastAPI backend, zero-dependency HTML/CSS/JS frontend,
single command to run.

> Built as part of a material science project on automated steel surface
> defect detection.

## ✨ Features

- **Drag-and-drop inspection** — drop or browse a surface image, get
  results instantly, no page reloads
- **Live bounding-box overlay** — detections drawn directly on the image,
  color-coded per defect class
- **Inspection tag panel** — per-class counts, total defect count, and a
  clickable class list that highlights just that defect type on the image
- **Single-service deployment** — the FastAPI backend serves the frontend
  itself; one command starts the whole app
- **Live health indicator** — the UI polls the backend and shows
  connected/disconnected status in real time


## 🧱 Tech Stack

| Layer      | Tech                              |
|------------|------------------------------------|
| Detection  | YOLO (Ultralytics)                 |
| Backend    | FastAPI, Python                    |
| Frontend   | Vanilla HTML / CSS / JS (no build step) |
| Inference  | PyTorch (CPU or CUDA)              |

## 📁 Project Structure

```
steel-defect-inspector/
├── app.py             FastAPI server — loads the model, exposes /api/predict, serves the frontend
├── index.html          drag-and-drop UI (single file: HTML + CSS + JS)
├── requirements.txt
├── best.pt             ← your trained YOLO weights (not included, see below)
└── README.md
```

## 🚀 Quick Start

```bash
git clone https://github.com/<your-username>/steel-defect-inspector.git
cd steel-defect-inspector

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Add your trained weights as `best.pt` in the project root (see below if
you don't have one yet), then run:

```bash
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** — the backend serves the UI directly, so
there's nothing else to start.

> **GPU inference**: if you have a CUDA GPU, install the matching PyTorch
> build *before* `ultralytics` for accelerated inference — see
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
> It runs fine on CPU otherwise, just slower per image.


```

## 🔍 How It Works

1. The frontend posts the dropped image to `POST /api/predict`.
2. The backend runs the YOLO model at a low confidence floor
   (`MIN_CONFIDENCE = 0.05` in `app.py`) and returns every detection above
   that floor — class, confidence, and pixel bounding box — along with
   per-class counts and inference time.
3. The frontend applies its own display cutoff (`DISPLAY_THRESHOLD = 0.25`
   in `index.html`) before drawing boxes or counting defects, so the
   cutoff can be retuned without re-running the model.
4. The inspection tag panel groups detections by class; clicking a row
   highlights just that class's boxes on the image.

## 📡 API Reference

**`GET /api/health`**
```json
{ "status": "ok", "model_file": "best.pt", "classes": ["scratch", "pitted_surface", "..."] }
```

**`POST /api/predict`** — multipart form, field name `file` (image)
```json
{
  "image_width": 1024,
  "image_height": 768,
  "inference_ms": 42,
  "detections": [
    { "class_name": "scratch", "confidence": 0.87, "bbox": [x1, y1, x2, y2] }
  ],
  "summary": { "scratch": 2, "pitted_surface": 1 },
  "total_defects": 3
}
```

## 🛠️ Customizing

- **Backend confidence floor** — `MIN_CONFIDENCE` in `app.py`; boxes below
  this never reach the frontend at all.
- **Frontend display cutoff** — `DISPLAY_THRESHOLD` in `index.html`'s
  `<script>` block.
- **Class colors** — the `PALETTE` array near the top of the same
  `<script>` block.
- **CORS** — `app.py` currently allows any origin (`allow_origins=["*"]`)
  for easy local dev; restrict it before deploying somewhere public.
- **Calling the API from a separately hosted frontend** — set `API_BASE`
  near the top of `index.html`'s `<script>` block to the backend's full
  URL.

## 🩺 Troubleshooting

| Issue | Fix |
|---|---|
| "server unreachable" in the header | Backend isn't running, or isn't reachable at `API_BASE` in `index.html`. Confirm `uvicorn` is running and you're on `http://localhost:8000`. |
| Startup error: "No model found at ..." | `best.pt` isn't in the project root (or wherever `MODEL_PATH` points). |
| UI doesn't load, only the API works | `FRONTEND_DIR` doesn't point at the folder containing `index.html`. |
| Slow first prediction | Ultralytics fuses/warms up the model on its first call; later predictions are faster. |

