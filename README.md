# SandhiSetu

**Integrated AI Platform for Knee Osteoarthritis Screening**

Team Apex · Smart India Hackathon 2026 · PS# SIH26004 · Theme: MedTech / BioTech / HealthTech

A modular Streamlit application for preliminary osteoarthritis (OA) screening in primary healthcare centres, rural health camps, and community outreach programs — built for regions like the North Eastern Region (NER) where a single diagnostic tool (X-ray, trained physiotherapist, gait lab) often isn't available on-site.

---

## Table of Contents

- [Three Independent Screening Pathways](#three-independent-screening-pathways)
- [Setup](#setup)
- [Hardware Module](#hardware-module)
- [Wiring Up the Real G5 Model](#wiring-up-the-real-g5-model)
- [AI Posture Detection](#ai-posture-detection)
- [Updating the Hardware Rule Thresholds](#updating-the-hardware-rule-thresholds)
- [Project Structure](#project-structure)
- [Important Notes](#important-notes)

---

## Three Independent Screening Pathways

| | Hardware OA Risk Marker | Software OA Risk Marker | AI Posture Detection |
|---|---|---|---|
| Input | Patient info + history + ESP32 sensor data | Patient info + history + knee X-ray image | Patient info + history + lower-limb photo |
| Method | **Rule/threshold-based engine** — no ML/AI | **G5 deep-learning model** (image classification AI) | **MediaPipe pose-landmark model** (AI) + rule-based angle thresholds |
| Output | Hardware OA Risk Score / Screening Category (Low / Moderate / High) | Joint Problem Detected (Yes/No), OA probability, Grad-CAM, Score-CAM, LIME | Hip-Knee-Ankle (HKA) angle per leg, Varus/Valgus classification, asymmetry flag |
| Report | Hardware OA Screening Report (PDF) | Software OA Screening Report (PDF) | AI Posture Detection Report (PDF) |

The three pathways are never combined, averaged, or turned into a single overall probability — they are fully independent by design, each with its own result screen and its own downloadable PDF report. A health worker screens with whatever is available on-site.

> **Where does AI fit in each pathway?** Hardware uses no ML/AI at all. Software's G5 model directly classifies the X-ray as Normal/OA. AI Posture Detection uses a pre-trained AI model (MediaPipe) only to *locate* hip/knee/ankle landmarks in the photo — the risk flag itself (varus/valgus, asymmetry) is then a plain threshold comparison on the measured angle, the same style of rule engine as the Hardware pathway.

---

## Setup

```bash
git clone <this-repo-url>
cd oa_screening_app

pip install -r requirements.txt

# Install ONE deep-learning backend matching your G5 model file format:
pip install torch          # for a .pt / .pth G5 model
# or
pip install tensorflow tf-keras   # for the supplied .h5 EfficientNetB0 G5 model

# AI Posture Detection needs MediaPipe (already in requirements.txt):
# pip install mediapipe

streamlit run app.py
```

> **Why `tf-keras` too?** The supplied `final_model.h5` was trained and saved with TF 2.15's Keras 2 engine. Modern TensorFlow ships Keras 3 by default, which cannot correctly rebuild this model's nested Functional graph (you'll see errors like `DepthwiseConv2D ... unrecognized keyword 'groups'` or `Invalid Functional model configuration ... loops or disconnected nodes` if `tf-keras` is missing). Installing `tf-keras` and letting `utils/model_loader.py` set `TF_USE_LEGACY_KERAS=1` (already wired in) loads the model with the same Keras 2 engine that wrote it, sidestepping the incompatibility entirely.

**Requirements:** Python 3.9–3.11 (MediaPipe does not currently publish wheels for every Python version).

---

## Hardware Module

The Hardware OA Risk Marker pathway is built around a low-cost wearable sensor unit — no ML/AI runs on this pathway, every result is a plain threshold comparison (see [Updating the Hardware Rule Thresholds](#updating-the-hardware-rule-thresholds)).

### Components

| Component | Role |
|---|---|
| ESP32 DevKit V1 | Main controller — Wi-Fi/BLE enabled, reads all sensor channels and streams them to the app |
| 3× MPU6050 (IMU) | Knee ROM sensor (flexion/extension range of motion) + 2× gait sensors (thigh/shin) for walking cadence and gait-cycle asymmetry |
| TCA9548A I2C Multiplexer | All three MPU6050 units share the same I2C address, so the multiplexer switches between them on separate channels |
| 2× FSR (Force-Sensitive Resistors) | Insole-mounted, one per foot — measure left-right load-bearing (plantar pressure) asymmetry |

### Wiring

- Each MPU6050 connects to its own TCA9548A channel (SDA/SCL), rather than sharing the I2C bus directly, since all three units default to the same I2C address.
- The two FSR sensors connect to ESP32 analog input pins (via a simple voltage-divider circuit).
- ESP32 streams all four readings (Knee ROM, cadence, gait asymmetry, pressure asymmetry) over Wi-Fi/BLE to the app.

### Circuit Simulation (Tinkercad)

Before physical assembly, the full sensor circuit was simulated in **Tinkercad Circuits** to validate the wiring (ESP32 ↔ TCA9548A ↔ 3× MPU6050, plus the FSR voltage-divider inputs) and confirm the I2C multiplexing logic addresses each IMU correctly before committing to hardware.

### Live vs. Demonstration Mode

If no ESP32 is physically connected, the app's Hardware pathway falls back to a **Demonstration / Manual Input Mode**, where the four sensor values are entered manually — useful for testing the risk engine, PDF report generation, and UI without the physical unit present. The live-device serial/BLE/HTTP reader is a wiring point in `components/hardware_module.py` (`render_hardware_module()`), ready to be connected to the actual ESP32 firmware's data stream.

---

## Wiring Up the Real G5 Model

The Software OA Risk Marker will not fabricate a result. Until a real, compatible model is supplied it will show a clear "model not available" message.

`config/settings.py` is **already pre-configured** to match the EfficientNetB0-based Keras model described in the training notebook (`model.save("final_model.h5")`):

| Setting | Value | Why |
|---|---|---|
| `G5_MODEL_PATH` | `models/final_model.h5` | drop your `.h5` file here |
| `G5_MODEL_BACKEND` | `auto` → detects TensorFlow from `.h5` | |
| `G5_CLASS_LABELS` | `Normal,Osteoarthritis` | matches `train_ds_raw.class_names` |
| `G5_POSITIVE_CLASS_INDEX` | `1` | index of `Osteoarthritis` |
| `G5_PREPROCESS_MODE` | `none` (raw 0-255 pixels) | Keras's `efficientnet.preprocess_input()` is an identity function — normalization is built into the model |
| `G5_GRADCAM_BASE_LAYER_NAME` | `efficientnetb0` | the backbone is a **nested** sub-model, not a flat set of layers |
| `G5_GRADCAM_TARGET_LAYER` | `top_conv` | EfficientNetB0's final conv layer (also used by Score-CAM) |
| `G5_SCORECAM_MAX_CHANNELS` | `64` | number of backbone channels Score-CAM re-runs the model on (32=faster, 128=slower, more detailed) |
| `G5_LIME_NUM_SAMPLES` / `G5_LIME_NUM_FEATURES` | `300` / `10` | LIME perturbation samples / highlighted super-pixel regions (CPU-practical defaults) |

**All you need to do:** place `final_model.h5` (or `best_model.h5`) into the `models/` folder. Everything else already matches.

If you retrain with a different architecture, class order, or preprocessing, update the corresponding environment variable (or edit `config/settings.py` directly):
- `G5_MODEL_PATH` / `G5_MODEL_BACKEND` — model file location and format (`pytorch`, `tensorflow`, or `auto`)
- `G5_CLASS_LABELS` / `G5_POSITIVE_CLASS_INDEX` — must match the model's actual output order
- `G5_PREPROCESS_MODE` — `none` (raw pixels), `rescale_0_1` (÷255), or `imagenet_normalize` (÷255 + mean/std) — use whichever matches how the model was trained
- `G5_GRADCAM_BASE_LAYER_NAME` — leave empty if your model is a flat (non-nested) Keras model; Grad-CAM will then look for `G5_GRADCAM_TARGET_LAYER` directly on the top-level model
- `G5_GRADCAM_TARGET_LAYER` — name (or substring) of the target conv layer
- If a PyTorch model file only contains a `state_dict` (weights only, no architecture), add the actual G5 model class definition into `utils/model_loader.py::_load_pytorch_model` — this is called out with a clear error message if it's ever hit.

Nothing in `predictor.py`, `gradcam.py`, `scorecam.py`, or `lime_explain.py` invents numbers: if the model can't be loaded, or inference/Grad-CAM/Score-CAM/LIME fails, the app reports the exact reason instead of a fake result — each of the three explainability methods fails independently, so e.g. a Score-CAM error never blocks the prediction, Grad-CAM, or LIME from still being shown.

### Nested EfficientNet architecture (Grad-CAM & Score-CAM)

Because the training notebook wraps `EfficientNetB0` as a **nested** sub-model (a layer literally named `"efficientnetb0"` inside the outer classification model), a plain `model.get_layer("top_conv")` call would fail — that layer lives *inside* the backbone, not on the outer model. `utils/tf_architecture.py` is the ONE place that discovery logic lives (used by both Grad-CAM and Score-CAM):
1. try `G5_GRADCAM_BASE_LAYER_NAME` / `G5_GRADCAM_TARGET_LAYER` exactly,
2. fall back to a substring match,
3. fall back to the last layer with a 4D output shape (architecture-agnostic).

`utils/gradcam.py` then builds a grad-model from the backbone's own input to `(top_conv output, backbone output)` and manually replays the classification head (`GlobalAveragePooling2D → Dropout → Dense`) inside the gradient tape — exactly mirroring the training notebook's `get_grad_cam()`. `utils/scorecam.py` reuses the same discovery to mask the input with the backbone's strongest activation channels and re-runs the full model on each masked image (no gradients needed), capped at `G5_SCORECAM_MAX_CHANNELS` channels to stay CPU-practical.

`utils/heatmap_utils.py` also provides `get_high_activation_regions()`, which thresholds the Grad-CAM heatmap and draws bounding boxes around the resulting regions — shown as a bonus visual in the app (not included in the PDF, to keep the report a manageable size).

### LIME

`utils/lime_explain.py` perturbs super-pixels of the resized X-ray, asks the real model to predict on each perturbation, and returns two overlays — "positive regions" (support the actual predicted class) and "positive + negative" (green = supports, red = opposes). It always explains the same class the rest of the page is showing (the model's actual prediction), so Grad-CAM/Score-CAM/LIME are never explaining three different classes. Requires the `lime` and `scikit-image` packages (in `requirements.txt`).

All three explainability outputs (Grad-CAM, Score-CAM, LIME) appear side-by-side in the app and are all included in the downloadable Software OA Screening Report PDF, laid out 3-per-row.

### Performance

The Software module shows a percentage progress bar while it works through preprocessing (10%) → model load (20%, cached after the first run) → inference (35%) → Grad-CAM (50%) → Score-CAM (70%, the slowest step — it re-runs the full model once per selected channel) → LIME (90%) → done (100%). The model is loaded once per app session (`@st.cache_resource` in `components/software_module.py`) rather than re-read from disk on every click. If Score-CAM or LIME feel too slow on CPU, lower `G5_SCORECAM_MAX_CHANNELS` and/or `G5_LIME_NUM_SAMPLES`.

### Standalone debug scripts

Two minimal scripts outside the full app let you confirm the model loads and predicts correctly in isolation, without patient forms or PDF generation getting in the way:
- `test_model.py path/to/xray.png` — command-line, prints the prediction and saves a Grad-CAM overlay image.
- `streamlit run streamlit_test_app.py` — a one-page Streamlit app: model path + upload + predict + Grad-CAM.

Useful for isolating a model-loading or architecture issue before re-testing inside the full multi-page app.

---

## AI Posture Detection

`utils/posture_analysis.py` runs Google's MediaPipe Pose model (`static_image_mode=True, model_complexity=2`) on the uploaded photo to locate the hip, knee and ankle landmarks for both legs — this is the pathway's only AI component. It then computes the Hip-Knee-Ankle (HKA) angle for each leg with plain trigonometry (180° = perfectly straight alignment) and compares it against `config/thresholds.py`: `POSTURE_KNEE_ANGLE_THRESHOLDS` (normal range, default 175°-185°) and `POSTURE_ASYMMETRY_MAX_DEGREES` (default 5°) for left-right asymmetry. An angle below the range is classified Varus, above it Valgus. Two visibility thresholds control robustness on partially-obscured photos: `POSTURE_LEG_VISIBILITY_THRESHOLD` (stricter, only affects what's drawn on the overlay) and `POSTURE_CORE_JOINT_MIN_VISIBILITY` (more lenient, used for the actual angle calculation — hip/knee/ankle are essential). If no pose is detected, or a required joint isn't visible enough, the module reports that plainly rather than fabricating an angle.

`components/posture_module.py` handles the upload UI and result display; `generate_posture_pdf_report()` in `utils/pdf_report.py` builds its independent PDF, embedding the annotated overlay image.

---

## Updating the Hardware Rule Thresholds

All sensor/history thresholds and scoring weights live in `config/thresholds.py`, fully separated from the engine logic in `utils/hardware_risk_engine.py`. Update the numbers there (with a citation in the comment) as better-validated clinical/research evidence becomes available — no code changes required elsewhere. The AI Posture Detection thresholds described above live in the same file.

---

## Project Structure

Full repository file tree, as it appears in VS Code's Explorer:

```
oa_screening_app/
├── app.py                       # Main Streamlit app: page config, navigation, 4-step workflow router
├── requirements.txt             # Python dependencies
├── README.md
├── test_model.py                # Standalone CLI: load -> predict -> Grad-CAM on one image
├── streamlit_test_app.py        # Standalone minimal Streamlit test app (no patient forms)
│
├── components/                  # UI + orchestration layer (one module per screening pathway)
│   ├── __init__.py
│   ├── patient_form.py          # Basic info + history intake forms (shared by all 3 pathways)
│   ├── hardware_module.py       # Hardware pathway UI (sensor input -> risk engine -> report)
│   ├── software_module.py       # Software pathway UI (X-ray -> G5 model -> Grad-CAM/Score-CAM/LIME -> report)
│   └── posture_module.py        # AI Posture Detection pathway UI (photo -> MediaPipe -> HKA angle -> report)
│
├── utils/                       # Core logic layer — no Streamlit imports, independently testable
│   ├── __init__.py
│   ├── patient_utils.py         # calculate_bmi(), classify_bmi(), input validation
│   ├── hardware_risk_engine.py  # process_sensor_data(), evaluate_hardware_risk() — pure rule engine, no ML
│   ├── image_utils.py           # X-ray upload validation + preprocessing
│   ├── model_loader.py          # load_image_model() — loads the G5 model (PyTorch or TensorFlow)
│   ├── predictor.py             # predict_image() — runs inference on the loaded G5 model
│   ├── tf_architecture.py       # Shared nested-backbone / conv-layer discovery (Grad-CAM + Score-CAM)
│   ├── gradcam.py               # generate_gradcam()
│   ├── scorecam.py              # generate_scorecam()
│   ├── lime_explain.py          # generate_lime_explanation()
│   ├── heatmap_utils.py         # overlay_heatmap(), get_high_activation_regions()
│   ├── posture_analysis.py      # analyze_posture_image() — MediaPipe pose detection + HKA angle + thresholds
│   ├── pdf_report.py            # generate_hardware/software/posture_pdf_report() — independent PDF builders
│   └── ui_helpers.py            # Shared HTML/CSS dashboard components (cards, badges, risk banners)
│
├── config/                      # Configuration — no logic, just constants
│   ├── __init__.py
│   ├── settings.py              # Paths, G5 model config, branding, medical disclaimer text
│   └── thresholds.py            # All clinical thresholds & risk-score weights (Hardware + Posture)
│
├── assets/
│   ├── style.css                # Healthcare dashboard theme (colors, cards, badges, layout)
│   └── temp/                    # Runtime scratch space for report-embedded images (auto-cleaned)
│
├── models/                      # Place the trained G5 model file here (final_model.h5 / .pt)
└── reports/                     # (optional) local copies of generated PDF reports
```

---

## Important Notes

- The Hardware OA Risk Marker contains **no machine learning** of any kind — every result can be traced back to a specific threshold comparison, shown in the "Detected Risk Markers" table.
- The Software OA Risk Marker's probability, class label, and Grad-CAM / Score-CAM / LIME visualizations all come directly from the real G5 model — never fabricated.
- The AI Posture Detection pathway uses MediaPipe only to locate landmarks; the Varus/Valgus classification and asymmetry flag are a plain, transparent threshold comparison on the measured angle, exactly like the Hardware pathway's rule engine.
- This is a **preliminary AI-assisted / rule-based screening tool**, not a diagnostic device. Every report carries this disclaimer.

---

## Team

Built by **Team Apex** for Smart India Hackathon 2026 — Problem Statement SIH26004 (Ministry of Development of North Eastern Region), Theme: MedTech / BioTech / HealthTech.
