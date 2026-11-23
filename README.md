# SolarGuard

IoT ve Sayısal Görüntü İşleme ile Güneş Paneli Kirlilik ve Verim Takip Sistemi — classical image processing only, no AI. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/get_dataset.py      # Kaggle dataset -> data/raw (no token needed)
```

## Run

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preview_panel.py    # FR1 visual check -> out/panel_preview.jpg
```
