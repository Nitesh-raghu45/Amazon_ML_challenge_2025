"""
train_price_model.py — Price prediction model training.

Pipeline:
  1. Load CLIP embeddings + extracted structured features
  2. Encode categorical features
  3. Train XGBoost + LightGBM models
  4. Evaluate with RMSLE (Amazon ML 2025 metric)
  5. Generate submission CSV
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from config import (
    EMBEDDING_DIR, EXTRACTION_DIR, ARCHIVE_DIR, OUTPUT_DIR,
    XGB_PARAMS, LGBM_PARAMS, RANDOM_STATE, TEST_SIZE
)


# ── Metric ────────────────────────────────────────────────────────────────────

def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error (Amazon ML 2025 competition metric)."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_features(split: str = "train") -> pd.DataFrame:
    """
    Load and merge embedding features with extracted structured features.

    Args:
        split: 'train', 'test', or 'sample_test'

    Returns:
        Merged DataFrame ready for model training.
    """
    # Load embedding CSV (if available)
    emb_path = EMBEDDING_DIR / split / "embedded.csv"
    if emb_path.exists():
        df = pd.read_csv(emb_path)
        print(f"Loaded embeddings: {df.shape}")
    else:
        # Fall back to raw CSV
        raw_path = ARCHIVE_DIR / f"{split}.csv"
        df = pd.read_csv(raw_path)
        print(f"No embeddings found. Loaded raw: {df.shape}")

    # Load extracted structured features (if available)
    extraction_path = EXTRACTION_DIR / f"{split}_extracted.csv"
    if extraction_path.exists():
        ext_df = pd.read_csv(extraction_path)
        ext_cols = [c for c in ext_df.columns if c != "sample_id"]
        df = df.merge(ext_df[["sample_id"] + ext_cols], on="sample_id", how="left")
        print(f"Merged extracted features: {df.shape}")

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "category", "product_form", "declared_unit", "packaging_type",
    "packaging_material", "target_demographic", "country_of_origin",
]

BINARY_COLS = [
    "edible", "is_premium", "is_bundle_deal", "is_limited_edition",
    "is_organic", "is_non_gmo", "is_gluten_free", "is_natural",
    "is_keto", "is_high_protein", "is_cruelty_free", "is_vegan",
]

NUMERIC_COLS = [
    "declared_quantity", "quantity_in_grams", "servings_per_container",
]

ARRAY_COLS = [
    "certifications_0", "certifications_1", "certifications_2",
    "allergens_contains_0", "allergens_contains_1", "allergens_contains_2",
    "top_ingredients_0", "top_ingredients_1", "top_ingredients_2",
]


def engineer_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """
    Encode categorical features and assemble feature matrix.

    Args:
        df: Input DataFrame.
        encoders: Pre-fitted LabelEncoders (for test set). None = fit new.
        fit: Whether to fit new encoders.

    Returns:
        (X: np.ndarray, encoders: dict)
    """
    df = df.copy()
    feature_cols = []
    if encoders is None:
        encoders = {}

    # Embedding columns
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    feature_cols.extend(emb_cols)

    # Categorical encoding
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "unknown")
                if "unknown" not in known:
                    le.classes_ = np.append(le.classes_, "unknown")
                df[col + "_enc"] = le.transform(df[col])
            else:
                df[col + "_enc"] = 0
        feature_cols.append(col + "_enc")

    # Binary columns
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            feature_cols.append(col)

    # Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            feature_cols.append(col)

    # Array-derived columns
    for col in ARRAY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            if fit:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col])
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le:
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else "unknown")
                    if "unknown" not in known:
                        le.classes_ = np.append(le.classes_, "unknown")
                    df[col + "_enc"] = le.transform(df[col])
                else:
                    df[col + "_enc"] = 0
            feature_cols.append(col + "_enc")

    # Keep only columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values.astype(np.float32)
    return X, encoders, feature_cols


def parse_price(price_str) -> float:
    """Parse price strings like '₹199', '$14.5', '25.0' to float."""
    if pd.isna(price_str):
        return np.nan
    s = str(price_str).strip()
    s = s.replace("₹", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# ── Model Training ────────────────────────────────────────────────────────────

def train_models(
    X_train, y_train, X_val, y_val,
    use_xgb: bool = True,
    use_lgbm: bool = True,
) -> dict:
    """
    Train XGBoost and/or LightGBM models.

    Returns:
        dict of {model_name: trained_model}
    """
    models = {}

    if use_xgb:
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(
            X_train, np.log1p(y_train),
            eval_set=[(X_val, np.log1p(y_val))],
            verbose=50,
        )
        val_pred = np.expm1(xgb_model.predict(X_val))
        val_pred = np.clip(val_pred, 0, None)
        score = rmsle(y_val, val_pred)
        print(f"XGBoost Validation RMSLE: {score:.4f}")
        models["xgboost"] = (xgb_model, score)

    if use_lgbm:
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgb_model.fit(
            X_train, np.log1p(y_train),
            eval_set=[(X_val, np.log1p(y_val))],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
        )
        val_pred = np.expm1(lgb_model.predict(X_val))
        val_pred = np.clip(val_pred, 0, None)
        score = rmsle(y_val, val_pred)
        print(f"LightGBM Validation RMSLE: {score:.4f}")
        models["lightgbm"] = (lgb_model, score)

    return models


# ── Feature Importance Plot ───────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols: list, model_name: str, top_n: int = 30):
    """Plot and save top-N feature importances."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    feat_imp = pd.Series(importances, index=feature_cols[:len(importances)])
    feat_imp = feat_imp.nlargest(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
    plt.title(f"{model_name} — Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = OUTPUT_DIR / f"{model_name}_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Amazon ML 2025 — Price Prediction Model Training")
    print("=" * 60)

    # 1. Load features
    train_df = load_features("train")

    # 2. Parse target
    if "price" in train_df.columns:
        train_df["price_num"] = train_df["price"].apply(parse_price)
    else:
        raise ValueError("'price' column not found in training data.")

    train_df = train_df.dropna(subset=["price_num"])
    train_df = train_df[train_df["price_num"] > 0]
    print(f"Training samples after filtering: {len(train_df)}")

    y = train_df["price_num"].values

    # 3. Feature engineering
    X, encoders, feature_cols = engineer_features(train_df, fit=True)
    print(f"Feature matrix shape: {X.shape}")

    # 4. Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 5. Train models
    models = train_models(X_train, y_train, X_val, y_val)

    # 6. Save models & encoders
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import pickle

    for name, (model, score) in models.items():
        model_path = OUTPUT_DIR / f"{name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {name} model: {model_path} (Val RMSLE: {score:.4f})")

        plot_feature_importance(model, feature_cols, name)

    enc_path = OUTPUT_DIR / "encoders.pkl"
    with open(enc_path, "wb") as f:
        pickle.dump({"encoders": encoders, "feature_cols": feature_cols}, f)
    print(f"Saved encoders: {enc_path}")

    # 7. Generate test predictions (ensemble)
    print("\nGenerating test predictions...")
    test_df = load_features("test")
    X_test, _, _ = engineer_features(test_df, encoders=encoders, fit=False)

    preds = []
    weights = []
    for name, (model, score) in models.items():
        pred = np.expm1(model.predict(X_test))
        pred = np.clip(pred, 0, None)
        preds.append(pred)
        weights.append(1.0 / (score + 1e-8))  # inverse RMSLE weighting

    # Weighted ensemble
    total_weight = sum(weights)
    ensemble_pred = sum(p * w for p, w in zip(preds, weights)) / total_weight

    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": np.round(ensemble_pred, 2),
    })

    sub_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission saved: {sub_path}")
    print(submission.head())

    # Validation scores summary
    print("\n" + "=" * 40)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 40)
    for name, (_, score) in models.items():
        print(f"  {name:<15} RMSLE: {score:.4f}")


if __name__ == "__main__":
    main()
