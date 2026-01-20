
"""
run_regressions_zscore_with_minmax_adapter.py

Goal
- Train all regression equations on Z-SCORED predictors (stable long-term).
- ALSO store the embedding MinMax stats (min/range) for the same predictors so the
  agent demo can convert:
      x_mm (0..1 from embedding) -> x_raw -> x_z (for regression) -> y_pred
- The demo should NOT “convert back into MinMax” for prediction. You only need:
      minmax -> raw -> zscore -> predict
  Then map predicted targets into your internal [-2,2] (or whatever) as you already do.

Outputs
- regression_analysis/linear_equation_pack.json

Notes
- Uses LinearRegression by default. If you want better generalization with n~37,
  swap LinearRegression -> Ridge(alpha=1.0) in two places.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # swap to Ridge for stability if desired
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------- CONFIG ----------------

DATA_PATH = "data_in_csv/state_space/s_total_features.csv"
FEATURE_NAMES_PATH = "data_in_csv/state_space/feature_names.csv"
RAW_FEATURES_OUT = "data_in_csv/21_set_raw.csv"
PACK_PATH = "regression_analysis/linear_equation_pack.json"

BEHAVIORS = [
    "Work", "Focused Learning", "Skill practicing", "Physical Endeavors",
    "Scrolling", "jorking", "Passive media", "Active Media",
    "System Architecture"
]

# substances / exogenous context inputs (keep as you had)
EXTRA_PREDICTORS = [ "NIC", "CAF"]

PREDICTORS_A = ["hours_slept", "bedtime_std_4_days", "Waking_Energy_Yesterday"] + EXTRA_PREDICTORS
PREDICTORS_B = BEHAVIORS + EXTRA_PREDICTORS

TARGETS_A = ["Waking Energy", "Sleep Quality"]
TARGETS_B = ["hours_slept", "bedtime_std_4_days", "Waking_Energy_Yesterday"]

# Regression normalization choice: "zscore" or None
REG_NORM = "zscore"

# Embedding normalization choice for agent feature-means: "minmax"
EMB_NORM = "minmax"

RANDOM_STATE = 42

# ----------------------------------------


@dataclass
class LinearEq:
    intercept: float
    coef: Dict[str, float]
    sigma: float

    # Regression-time normalizer metadata (for converting RAW -> REG space)
    norm: Optional[str] = None  # "zscore" or "minmax" or None
    mean: Optional[Dict[str, float]] = None
    scale: Optional[Dict[str, float]] = None
    min: Optional[Dict[str, float]] = None
    range: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "intercept": self.intercept,
            "coef": self.coef,
            "sigma": self.sigma,
            "norm": self.norm,
            "mean": self.mean,
            "scale": self.scale,
            "min": self.min,
            "range": self.range,
        }


@dataclass
class TransitionModelPack:
    # Equations
    eq_hours: LinearEq
    eq_bedstd: LinearEq
    eq_energy: LinearEq
    eq_sq: LinearEq
    eq_wake_yesterday: Optional[LinearEq] = None

    # Embedding MinMax stats for predictors (to map x_mm -> x_raw inside the demo)
    embedding_min: Optional[Dict[str, float]] = None
    embedding_range: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        out = {
            "eq_hours": self.eq_hours.to_dict(),
            "eq_bedstd": self.eq_bedstd.to_dict(),
            "eq_energy": self.eq_energy.to_dict(),
            "eq_sq": self.eq_sq.to_dict(),
            "embedding_min": self.embedding_min,
            "embedding_range": self.embedding_range,
        }
        if self.eq_wake_yesterday is not None:
            out["eq_wake_yesterday"] = self.eq_wake_yesterday.to_dict()
        return out



def compute_embedding_minmax_stats(df_features: pd.DataFrame, cols: List[str]) -> Dict:
    """
    Compute MinMax stats on the SAME feature table you use for embedding (after dropna),
    and return per-column min and range for conversion x_mm -> x_raw:

        x_raw = x_mm * range + min

    NOTE: This is the only mathematically correct way to convert MinMax values back to raw.
    """
    mins = df_features[cols].min(numeric_only=True)
    maxs = df_features[cols].max(numeric_only=True)
    ranges = maxs - mins
    emb_min = mins.to_dict()
    emb_range = ranges.to_dict()
    return emb_min, emb_range



def load_embedding_features(s_total_path: str, names_path: str) -> pd.DataFrame:
    names = pd.read_csv(names_path, header=None)[0].tolist()
    df = pd.read_csv(s_total_path, header=None)
    if len(names) != df.shape[1]:
        raise ValueError(f"Feature names count {len(names)} does not match columns {df.shape[1]}")
    df.columns = names
    return df


def fit_and_report_zscore(df: pd.DataFrame, predictors: List[str], target: str):
    X_raw = df[predictors].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    model = LinearRegression()  # swap to Ridge(alpha=1.0) if desired

    mean = None
    scale = None
    norm = None

    if REG_NORM == "zscore":
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        mean = dict(zip(predictors, scaler.mean_))
        scale = dict(zip(predictors, scaler.scale_))
        norm = "zscore"
    else:
        X = X_raw

    model.fit(X, y)
    r2 = float(model.score(X, y))
    n = int(len(df))
    p = int(len(predictors))
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / max(1, n - p - 1))

    # CV R^2 (pipeline so scaling happens inside folds)
    cv_r2_mean = np.nan
    cv_r2_std = np.nan
    if n >= 3:
        n_splits = min(5, n)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        if REG_NORM == "zscore":
            pipe = make_pipeline(StandardScaler(), LinearRegression())
            scores = cross_val_score(pipe, X_raw, y, cv=cv, scoring="r2")
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        cv_r2_mean = float(np.mean(scores))
        cv_r2_std = float(np.std(scores))

    coefs = dict(zip(predictors, model.coef_))
    residuals = y - model.predict(X)
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    return {
        "target": target,
        "n": n,
        "intercept": float(model.intercept_),
        "r2": r2,
        "adj_r2": adj_r2,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "coefs": coefs,
        "sigma": sigma,
        "norm": norm,
        "mean": mean,
        "scale": scale,
    }


def print_report(title: str, results: List[dict]) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for res in results:
        print(f"\nTarget: {res['target']}")
        print(f"n: {res['n']}")
        print(f"intercept: {res['intercept']:.6f}")
        print(f"R^2: {res['r2']:.6f}")
        print(f"Adj R^2: {res['adj_r2']:.6f}")
        if not np.isnan(res["cv_r2_mean"]):
            print(f"CV R^2 (mean ± std): {res['cv_r2_mean']:.6f} ± {res['cv_r2_std']:.6f}")
        print(f"residual sigma: {res['sigma']:.6f}")
        for name, val in res["coefs"].items():
            print(f"  {name}: {val:.6f}")


def main():
    df_features = load_embedding_features(DATA_PATH, FEATURE_NAMES_PATH)
   

    # Save raw feature table (your snippet)
    df_features.to_csv(RAW_FEATURES_OUT, index=False)

    # IMPORTANT: drop rows with incomplete rolling/shift context
    df_features = df_features.dropna().copy()

    # Compute embedding MinMax stats on the same df_features you embed/cluster on.
    # Include all predictors you might ever feed from the agent into regressions.
    all_predictors = sorted(list(set(PREDICTORS_A + PREDICTORS_B)))
    emb_min, emb_range = compute_embedding_minmax_stats(df_features, all_predictors)

    print(f"min value: {emb_min} range value : {emb_range}")
    # -------- Model A (context -> Energy, SQ) --------
    df_a = df_features[PREDICTORS_A + TARGETS_A].dropna().copy()
    results_a = [fit_and_report_zscore(df_a, PREDICTORS_A, t) for t in TARGETS_A]
    print_report("Model A: Sleep features -> Waking Energy & Sleep Quality", results_a)

    # -------- Model B (behaviors -> sleep-context) --------
    df_b = df_features[PREDICTORS_B + TARGETS_B].dropna().copy()
    results_b = [fit_and_report_zscore(df_b, PREDICTORS_B, t) for t in TARGETS_B]
    print_report("Model B: Behaviors -> Sleep-derived targets", results_b)

    # Build equations
    eq_energy = LinearEq(
        intercept=results_a[0]["intercept"],
        coef=results_a[0]["coefs"],
        sigma=results_a[0]["sigma"],
        norm=results_a[0]["norm"],
        mean=results_a[0]["mean"],
        scale=results_a[0]["scale"],
    )
    eq_sq = LinearEq(
        intercept=results_a[1]["intercept"],
        coef=results_a[1]["coefs"],
        sigma=results_a[1]["sigma"],
        norm=results_a[1]["norm"],
        mean=results_a[1]["mean"],
        scale=results_a[1]["scale"],
    )
    eq_hours = LinearEq(
        intercept=results_b[0]["intercept"],
        coef=results_b[0]["coefs"],
        sigma=results_b[0]["sigma"],
        norm=results_b[0]["norm"],
        mean=results_b[0]["mean"],
        scale=results_b[0]["scale"],
    )
    eq_bedstd = LinearEq(
        intercept=results_b[1]["intercept"],
        coef=results_b[1]["coefs"],
        sigma=results_b[1]["sigma"],
        norm=results_b[1]["norm"],
        mean=results_b[1]["mean"],
        scale=results_b[1]["scale"],
    )
    eq_wake_yesterday = LinearEq(
        intercept=results_b[2]["intercept"],
        coef=results_b[2]["coefs"],
        sigma=results_b[2]["sigma"],
        norm=results_b[2]["norm"],
        mean=results_b[2]["mean"],
        scale=results_b[2]["scale"],
    )

    pack = TransitionModelPack(
        eq_hours=eq_hours,
        eq_bedstd=eq_bedstd,
        eq_energy=eq_energy,
        eq_sq=eq_sq,
        eq_wake_yesterday=eq_wake_yesterday,
        embedding_min=emb_min,
        embedding_range=emb_range,
    )

    with open(PACK_PATH, "w", encoding="utf-8") as f:
        json.dump(pack.to_dict(), f, indent=2)

    print(f"\nSaved regression equation pack to {PACK_PATH}")
    print(f"Saved raw feature table to {RAW_FEATURES_OUT}")
    print("\nDemo-side conversion reminder:")
    print("  x_raw = x_mm * embedding_range[k] + embedding_min[k]")
    print("  x_z   = (x_raw - eq.mean[k]) / eq.scale[k]   (for zscore equations)")
    print("  y_pred = eq.intercept + sum(eq.coef[k] * x_z[k]) + noise")


if __name__ == "__main__":
    main()


# Non-linear modeling ideas (same X->Y variables):
# - Polynomial regression (add squared/cubic terms per predictor)
# - Interaction terms (X1*X2) for combined effects
# - Splines (e.g., cubic splines) for smooth non-linear fits
# - GAMs (generalized additive models): interpretable curves per feature
# - Random Forest / Gradient Boosting for flexible non-linear modeling
# - SVR with RBF kernel for smooth, non-linear decision surfaces
# - k-NN regression for local, non-linear fits
# - Simple transforms (log/sqrt) if relationships are monotonic but curved
