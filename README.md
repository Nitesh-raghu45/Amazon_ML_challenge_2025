# Amazon ML 2025 Product Price Prediction

> **Competition**: Amazon ML Challenge 2025  
> **Task**: Predict the price of Amazon products from product images + catalog text  
> **Metric**: RMSLE (Root Mean Squared Log Error)

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Architecture](#-project-architecture)
- [Pipeline Overview](#-pipeline-overview)
- [File Structure](#-file-structure)
- [Setup & Installation](#-setup--installation)
- [Quick Start](#-quick-start)
- [Running Individual Stages](#-running-individual-stages)
- [Models Used](#-models-used)
- [Key Design Decisions](#-key-design-decisions)
- [Bug Fixes](#-bug-fixes-from-original-code)
- [Results](#-results)
- [Contributing](#-contributing)

---

## Problem Statement

Given a product's:
- **Image** (product photo on Amazon)
- **Catalog text** (name, description, attributes)

Predict the **price** (in INR) as accurately as possible.

This is a **regression task** evaluated by RMSLE, which penalizes under-predictions more than over-predictions.

---

## Project Architecture

```
                ┌──────────────────────────────────────────────┐
                │             Amazon ML 2025 Pipeline           │
                └──────────────────────────────────────────────┘

 ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
 │  Dataset │───▶│  Downloader │───▶│  VLM Extract │───▶│  CLIP Embedder│
 │ (CSV)    │    │ (parallel)  │    │  (Ollama LLM)│    │(image+text)   │
 └──────────┘    └─────────────┘    └──────────────┘    └───────┬───────┘
                                            │                    │
                              Structured JSON              Embeddings
                              (category, form,            (1024-dim)
                               packaging, etc.)                  │
                                            │                    │
                                            └────────┬───────────┘
                                                     ▼
                                          ┌──────────────────┐
                                          │  Feature Matrix   │
                                          │  (embeddings +   │
                                          │  structured feats)│
                                          └────────┬─────────┘
                                                   ▼
                                     ┌─────────────────────────┐
                                     │  XGBoost + LightGBM     │
                                     │  Ensemble Prediction     │
                                     └─────────────────────────┘
                                                   ▼
                                           submission.csv
```

---

## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| **1. Download** | `downloader.py` | Downloads product images in parallel (256 threads) |
| **2. Extract** | `extractor.py` | VLM extracts structured JSON (category, packaging, etc.) |
| **3. Embed** | `embedder.py` | CLIP generates 1024-dim image+text embeddings |
| **4. Train** | `train_price_model.py` | XGBoost + LightGBM ensemble price prediction |
| **5. Submit** | `pipeline.py` | Generates `submission.csv` |

---

## File Structure

```
amezonML2025/
│
├── pipeline.py              #  Main orchestrator — run this for E2E
├── config.py                #   All paths, hyperparams, and model settings
│
├── downloader.py            #  Parallel image downloader
├── extractor.py             #  VLM-based structured feature extractor
├── embedder.py              #  CLIP image+text embedding generator
├── train_price_model.py     # XGBoost + LightGBM price predictor
│
├── schema.py                # Pydantic schema (ProductExtraction model)
├── prompts.py               # LLM system prompt builder
├── utils.py                 #  Shared utilities (logging, retry, timer)
│
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
│
├── archive/                 # Dataset CSVs (from Kaggle)
│   ├── train.csv
│   ├── test.csv
│   └── sample_test.csv
│
├── Images/                  # Downloaded product images
│   ├── train/
│   ├── test/
│   └── sample_test/
│
└── outputs/                 # Model outputs
    ├── submission.csv
    ├── xgboost_model.pkl
    ├── lightgbm_model.pkl
    ├── encoders.pkl
    └── *_feature_importance.png
```

> **Legacy files** (kept for reference):
> - `EmbeddingImageText.py` → replaced by `embedder.py`
> - `imageCaptionModel.py` → replaced by `extractor.py`
> - `imageCaptionModelUsingOllama.py` → merged into `extractor.py`
> - `SavingModelOutputJsonToCsv.py` → merged into `extractor.py`
> - `pydanticCheckJson.py` → replaced by `schema.py`
> - `SystemPrompt.py` → replaced by `prompts.py`
> - `download.py` → replaced by `downloader.py`

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended, 16GB+ VRAM for LLM extraction)
- [Ollama](https://ollama.com) installed and running (for VLM extraction)
- Dataset CSVs placed in `./archive/`

### 1. Clone / Navigate to the project

```bash
cd "amezonML2025"
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (NVIDIA CUDA 12.x)** — For cuDF/cuML GPU acceleration, uncomment the RAPIDS lines in `requirements.txt` and run:
> ```bash
> pip install --extra-index-url https://pypi.nvidia.com cudf-cu12==24.4.* cuml-cu12==24.4.*
> ```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your Ollama settings:

```env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=your-api-key-here
OLLAMA_MODEL=ministral-3:14b-cloud
```

### 5. Start Ollama (for VLM extraction)

```bash
ollama serve
ollama pull ministral-3:14b-cloud   # or your preferred multimodal model
```

### 6. Get the dataset

Download from the competition page and place in `./archive/`:
```
archive/
├── train.csv
├── test.csv
└── sample_test.csv
```

---

## Quick Start

### Run the complete pipeline end-to-end:

```bash
python pipeline.py
```

### Run with options:

```bash
# Skip image download (if already downloaded)
python pipeline.py --skip-download

# Only embed and train (skip download + extract)
python pipeline.py --skip-download --skip-extract

# Process only train split
python pipeline.py --splits train

# Extract starting from row 500 (resume)
python pipeline.py --stage extract --extract-start 500
```

---

## Running Individual Stages

### Stage 1: Download Images

```bash
python downloader.py
```

Downloads train, test, and sample_test images in parallel. Already-downloaded images are skipped automatically.

### Stage 2: Extract Structured Features (VLM)

```bash
python pipeline.py --stage extract --extract-batch-size 10
```

Sends each product image + catalog text to your Ollama VLM and extracts:
- Category, product form, packaging type/material
- Premium/bundle/limited edition flags
- Country of origin, manufacturer, certifications
- Declared quantity, unit, servings

Results saved to `archive/extraction/{split}/{start}-{end}.csv` (checkpointed).

### Stage 3: Generate CLIP Embeddings

```bash
python pipeline.py --stage embed
```

Generates 1024-dim embeddings (512 image + 512 text) using CLIP ViT-B/32.  
Output: `archive/embedding/{split}/embedded.csv`

### Stage 4: Train Price Prediction Model

```bash
python train_price_model.py
```

Trains XGBoost + LightGBM on the combined feature matrix. Saves:
- `outputs/submission.csv`
- `outputs/xgboost_model.pkl`
- `outputs/lightgbm_model.pkl`
- `outputs/encoders.pkl`
- Feature importance plots

---

## 🤖 Models Used

### 1. CLIP (openai/clip-vit-base-patch32)
- **Purpose**: Multimodal image+text embedding
- **Output**: 1024-dim vector (512 image + 512 text concatenated)
- **Why**: Captures visual product features (packaging, brand) + textual features jointly

### 2. Ollama VLM (ministral-3:14b-cloud)
- **Purpose**: Structured feature extraction from images + catalog text
- **Output**: JSON with 30+ product attributes
- **Why**: LLMs can read packaging text, infer category, and identify premium cues
- **Retry**: Up to 5 attempts with structured feedback on JSON/validation errors

### 3. XGBoost + LightGBM (Ensemble)
- **Purpose**: Price regression on the combined feature matrix
- **Target**: `log1p(price)` → prevents large price dominance
- **Ensemble**: Weighted by inverse RMSLE score
- **Features**: CLIP embeddings + encoded categorical + binary flags + numeric attrs

---

## 💡 Key Design Decisions

### Why CLIP embeddings?
Visual signals matter for pricing: premium packaging, brand logos, product size, and presentation all correlate with price. CLIP captures these without task-specific training.

### Why VLM extraction?
Raw catalog text is noisy and unstructured. A VLM can:
- Identify the category even from confusing text
- Read quantity from images (e.g., "500ml" visible on bottle)
- Detect premium branding cues

### Why log-transform the target?
Price has a right-skewed distribution (most items cheap, few very expensive). `log1p` normalization makes the regression task easier and aligns with the RMSLE metric.

### Why checkpoint extraction?
VLM extraction is slow (~1–5 seconds/row). Batched checkpointing means crashes don't lose progress — resume with `--extract-start`.

---

## Bug Fixes (from original code)

| File | Bug | Fix |
|------|-----|-----|
| `SavingModelOutputJsonToCsv.py` | `val[:3]` undefined variable | Fixed to `value[:3]` in `schema.py` `to_flat_dict()` |
| `SystemPrompt.py` | Uses undefined variable `j` for schema | `prompts.py` now calls `build_system_prompt(schema)` |
| `imageCaptionModel.py` | Missing `ProductExtraction` import | All imports unified in `extractor.py` → `schema.py` |
| `imageCaptionModelUsingOllama.py` | Hardcoded API key in source | Moved to `.env` / `config.py` via `os.getenv()` |
| `req.txt` | Contains pip command, not package list | Replaced with proper `requirements.txt` |
| `pydanticCheckJson.py` | Missing Group B-D fields from system prompt | Full schema in `schema.py` with all 30+ fields |

---

## Results

| Model | Validation RMSLE |
|-------|-----------------|
| XGBoost (embeddings only) | TBD |
| LightGBM (embeddings only) | TBD |
| XGBoost (embeddings + extracted) | TBD |
| LightGBM (embeddings + extracted) | TBD |
| **Ensemble (weighted)** | **TBD** |

> Run `python train_price_model.py` to populate these results.

---

## Security Notes

- **API keys** are loaded from environment variables (`.env`) — never commit them to git
- `.env` is included in `.gitignore` by default
- The `OLLAMA_API_KEY` in `imageCaptionModelUsingOllama.py` was hardcoded — moved to config

---

## References

- [Amazon ML Challenge 2025](https://www.amazon.science/blog)
- [CLIP Paper — Radford et al. 2021](https://arxiv.org/abs/2103.00020)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [LightGBM Documentation](https://lightgbm.readthedocs.io)
- [Ollama](https://ollama.com)
- [Pydantic v2 Docs](https://docs.pydantic.dev/latest/)

---

## Author

**Nitesh**  
Amazon ML Challenge 2025 Participant  

---

## License

This project is for competition/educational use only.  
Amazon dataset is subject to [Amazon's Terms of Service](https://www.amazon.com/gp/help/customer/display.html?nodeId=508088).
