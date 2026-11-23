# SolarGuard

IoT ve Sayısal Görüntü İşleme ile Güneş Paneli Kirlilik ve Verim Takip Sistemi — classical image processing only, no AI. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/get_dataset.py      # Kaggle dataset -> data/raw (no token needed)
.venv/bin/python scripts/eval_dataset.py     # Dataset Mode results -> out/eval.json
```

## Run the website

```bash
.venv/bin/uvicorn solarguard.web:app --port 8000
```

Open http://localhost:8000

| Page | What it shows |
|---|---|
| `/` Manager | 6 simulated panels (live camera frame, click for the dirt overlay), power, kWh/₺ lost, Manual/Auto mode, approvals, history chart, settings |
| `/staff` | Cleaning/inspection tasks (start → done) and notifications. Done cleans the virtual panel. |
| `/dataset` | Accuracy on the Kaggle set (dev/test split), confusion matrix, every image with its overlay |
| `/try` | Upload any photo and see each pipeline step |

Demo buttons on each panel ("dust", "birds") dirty a panel immediately.

## Pipeline (`solarguard/`)

| File | Step |
|---|---|
| `panel.py` | FR1 panel ROI: HSV colour + local-std texture → morphology → convex hull |
| `dirt.py` | FR2–3 bilateral filter + CLAHE, dirt index `V·(1−S)`, Otsu/fixed threshold, opening, elongated-component removal, glare mask |
| `analyze.py` | FR4–5 features (dirt %, snow area, rust colour, Canny density, white blobs, saturation, NumPy GLCM) + hand-written rules |
| `sim.py` | Digital twin: dust texture + alpha blending, droppings, daylight, power model |
| `live.py` | Decision engine (K readings in a row + cooldown), tasks, notifications, SQLite history |
| `web.py` | FastAPI + static pages in `web/` |

## Tests

```bash
.venv/bin/python -m pytest -q
```
