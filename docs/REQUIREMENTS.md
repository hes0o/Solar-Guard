# SolarGuard — Goals & Requirements

**SolarGuard: IoT ve Sayısal Görüntü İşleme ile Güneş Paneli Kirlilik ve Verim Takip Sistemi**

Güneş panellerinin yüzeyindeki tozu ve kirliliği görüntü işleme ile tespit edip, enerji kaybını önlemek için otomatik temizlik kararı vermek.

## 1. Constraints

| Constraint | Decision |
|---|---|
| Method | **Classical digital image processing only. No AI/ML** (no CNN, SVM, k-NN, or any trained model). Classification is hand-written rules on hand-crafted features. Thresholds are chosen by inspecting feature histograms. |
| Hardware | **None.** Fully software; the IoT side (camera + power sensor) is simulated. |
| Data | Kaggle *Solar Panel Images: Clean and Faulty* (6 classes). |
| Delivery | Web app reachable in a browser, hosted on our own (weak) server. Colab allowed for experiments/report. |

## 2. Goals

| # | Goal | Success measure |
|---|---|---|
| G1 | Locate the panel and rectify it to a frontal view | Panel ROI found in ≥ 90% of usable images |
| G2 | Segment dirt and compute **dirt %** | Visual check + IoU on a small hand-labeled subset |
| G3 | Rule-based classification into Temiz / Tozlu / Kuş pisliği / Kar / Anomali | Accuracy + confusion matrix on the Kaggle set (expected 65–80%, reported honestly) |
| G4 | Estimate energy & money loss from dirt | kWh and ₺ loss shown per panel |
| G5 | Make a stable cleaning decision | No flapping: K consecutive readings + cooldown |
| G6 | Web management panel with Manual and Auto modes, staff notifications | Full demo loop: dirt → decision → task → cleaned → panel resets |

## 3. Class mapping (Kaggle → SolarGuard)

| Kaggle class | SolarGuard label | Action |
|---|---|---|
| Clean | Temiz | none |
| Dusty | Tozlu | cleaning if over threshold |
| Bird-drop | Kuş pisliği | cleaning (spot) |
| Snow-Covered | Kar | wait / cleaning (configurable) |
| Electrical-damage, Physical-damage | Anomali | **inspection**, not cleaning |

## 4. Two modes, one pipeline

Both modes call the same `analyze(image) -> result` function.

- **Dataset Mode** — runs the pipeline on every Kaggle image and reports accuracy, the confusion matrix, and failure examples. This is the *evidence* for the image-processing part.
- **Live Mode (simulation / digital twin)** — N virtual panels. Each starts from a clean Kaggle image; a simulated clock adds dust (noise texture + blur + alpha blend), bird drops (soft white blobs), and day/night lighting changes. A "virtual camera" sends each panel's current frame to the pipeline every few seconds. This shows *how the system behaves* in operation.

> Note for the report: simulated power is derived from simulated dirt, so Live Mode is a demonstration, not evidence. Evidence comes from Dataset Mode.

## 5. Functional requirements

### Image-processing core
- **FR1** Panel localization: colour (blue / dark / grey cells, HSV) + texture (local std from grid lines) → morphology → large regions → convex hull, so dirt on the panel stays inside the ROI. (Dataset images are angled array shots, so a single-quad perspective warp was dropped.) Fallback: whole image if coverage < 10%.
- **FR2** Preprocessing: resize, CLAHE, denoise, glare/specular mask.
- **FR3** Dirt segmentation (HSV/Lab thresholds, Otsu/adaptive threshold, morphology) → mask + **dirt %**.
- **FR4** Feature extraction: color stats, GLCM texture, edge density, white-blob count/size.
- **FR5** Rule-based classifier → label + human-readable reason (e.g. "dirt 34%, low saturation, low edge density → Tozlu").

### Dataset Mode
- **FR6** Batch run over the dataset (background job) → metrics stored in DB.
- **FR7** Per-image step visualization (original → ROI → preprocessed → mask) for report and UI.
- **FR8** Single-image "try it" upload on the web for demos.

### Live Mode
- **FR9** N virtual panels with simulated clock, dust accumulation, bird drops, lighting.
- **FR10** Simulated power: `P = P_nominal × irradiance(t) × (1 − k·dirt%) + noise`; kWh and ₺ loss.
- **FR11** Decision engine: `dirt% > T1` for K consecutive readings → cleaning needed; cooldown after a task.

### Management panel (web)
- **FR12** Panel grid: status color, latest frame, mask, dirt %, power, trend chart.
- **FR13** **Manual mode**: system *suggests* a cleaning service; manager approves/rejects.
- **FR14** **Auto mode**: cleaning task created and sent to staff with no approval.
- **FR15** Staff view: in-app notifications + task list (Yeni → Başladı → Tamamlandı). Completing a task cleans the virtual panel.
- **FR16** Settings: thresholds, K, cooldown, mode switch, simulation speed.
- **FR17** History: every reading, decision and task stored (SQLite).

## 6. Non-functional requirements

- **NFR1** Runs on the project server (Pentium E5300, 2 cores, no AVX, ~1.3 GB free RAM): images resized to ≤ 640 px, GLCM on a 256 px / 32-level copy, target < 1 s per image, single worker, < 512 MB RAM. The Dataset Mode batch may run on Colab/laptop and its results be uploaded.
- **NFR2** One process: web server + simulation loop + SQLite. No external services.
- **NFR3** Each pipeline step is a pure function, testable and visualizable on its own.
- **NFR4** Dashboard refreshes live (polling every few seconds).
- **NFR5** Simple login with two roles: manager, staff.

## 7. Architecture

```
Browser (manager / staff)
   │  HTTP + polling
   ▼
FastAPI (one process, uvicorn)
 ├─ web pages (Jinja templates + small JS)
 ├─ REST API
 ├─ simulation loop (background task) ──► virtual camera frames
 ├─ pipeline: analyze(image)  ◄── also used by Dataset Mode job
 ├─ decision engine → tasks / notifications
 └─ SQLite + images on disk
```

**Stack:** Python, OpenCV (headless), NumPy, scikit-image (GLCM only), FastAPI, Jinja, SQLite, Chart.js. Deployed behind Caddy/nginx on our server.

## 8. Out of scope
- Real hardware (camera, sensors, cleaning robot)
- Any AI/ML model
- Reliable distinction between electrical and physical damage (both → Anomali)
- External notification channels (email/SMS/Telegram)

## 9. Milestones
1. Repo skeleton + dataset download + Colab exploration notebook
2. FR1–FR5 pipeline + Dataset Mode metrics (FR6–FR7)
3. Live simulation + decision engine (FR9–FR11)
4. Web panel: manager + staff views, Manual/Auto (FR12–FR17)
5. Deploy to server, write report, prepare demo
