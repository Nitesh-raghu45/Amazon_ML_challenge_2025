# Amazon ML 2025 - Implementation Plan

## Project Summary
Amazon ML Challenge 2025 — Product Price Prediction from Images + Catalog Text

Pipeline:
1. Download product images
2. Extract structured JSON from image + text using VLM (Qwen or Ollama)
3. Generate CLIP embeddings (image+text fusion)
4. Predict price using ML model

## Files to Create/Fix
- pipeline.py (orchestrator - full E2E)
- requirements.txt (proper pip format)
- config.py (centralized config)
- train_price_model.py (XGBoost/LightGBM on embeddings)
- README.md (comprehensive)

## Bugs Found
- SavingModelOutputJsonToCsv.py: line 14 `val[:3]` should be `value[:3]`
- imageCaptionModel.py: missing ProductExtraction import from pydanticCheckJson
- imageCaptionModelUsingOllama.py: missing ProductExtraction import, hardcoded API key exposed
- SystemPrompt.py: uses `j` variable without defining it (schema not injected)
- req.txt: not proper requirements.txt format (contains pip command, not packages)
