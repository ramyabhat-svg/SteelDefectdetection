# Steel Defect Inspector

A drag-and-drop UI for your YOLO steel defect model. Drop an image in, the
backend runs `best.pt` against it, and the frontend draws the bounding
boxes and a defect summary ("inspection tag") next to the image.

```
steel-defect-inspector/
├── backend/
│   ├── app.py            FastAPI server — loads best.pt, exposes /api/predict
│   ├── requirements.txt
│   └── best.pt            ← put your trained weights here (not included)
├── frontend/
│   └── index.html         drag-and-drop UI, served by the backend
└── README.md
```

## 1. Add your model

Copy your trained weights into the backend folder so the path is:

```
backend/best.pt
```

## 2. Install dependencies

A virtual environment is recommended:

```bash
cd steel-defect-inspector/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you're on a machine with a CUDA GPU, install the matching PyTorch build
*before* `ultralytics` for GPU-accelerated inference — see
https://pytorch.org/get-started/locally/. Otherwise it'll run fine on CPU,
just slower per image.

## 3. Run it

```bash
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000** — the backend serves the frontend
directly, so there's nothing else to start.

## How it works

- The frontend posts the dropped image to `POST /api/predict`.
- The backend runs `model.predict()` once at a low confidence floor (0.05)
  and returns every detection it found, along with each box's class,
  confidence, and pixel coordinates.
- The confidence slider in the UI filters those results client-side, so
  tightening or loosening the threshold is instant — no extra requests.
- The right-hand "inspection tag" panel groups detections by class,
  shows counts and average confidence per defect type, and stamps a
  PASS/FAIL verdict based on whether anything cleared the threshold.
- Clicking a defect row highlights its boxes on the image.

## Customizing

- **Confidence floor**: change `MIN_CONFIDENCE` in `backend/app.py` if you
  want the backend itself to discard very low-confidence boxes before
  they're ever sent to the browser.
- **Class colors**: edit the `PALETTE` array near the top of the `<script>`
  block in `frontend/index.html`.
- **CORS**: the backend currently allows requests from any origin
  (`allow_origins=["*"]`) for easy local development. If you deploy this
  somewhere reachable by others, restrict it to your actual frontend
  origin.
- **Deploying separately**: if you ever serve the frontend from somewhere
  other than the FastAPI app itself, set `API_BASE` near the top of the
  `<script>` block in `index.html` to your backend's full URL.

## Troubleshooting

- **"server unreachable" in the header** — the backend isn't running, or
  isn't reachable at the address the page expects. Confirm `uvicorn` is
  running and you're visiting `http://localhost:8000`.
- **Startup error about missing best.pt** — the server checks for the
  weights file on boot; make sure it's at `backend/best.pt` exactly.
- **Slow first prediction** — Ultralytics fuses/warms up the model on its
  first call; subsequent predictions are faster.
