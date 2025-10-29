from __future__ import annotations
import argparse, pathlib, joblib, mlflow
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from lib.common.settings import load_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="NBA")
    args = ap.parse_args()
    s = load_settings()

    wh = pathlib.Path(s.paths["warehouse"]) / args.league
    out_dir = pathlib.Path("artifacts") / args.league
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = wh / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} missing — run make features first.")

    df = pd.read_parquet(features_path)
    print(f"[train] Loaded {len(df):,} samples with labels (unique labels={df['label'].nunique()})")

    feature_cols = ["imp_prob_mean", "vig_spread", "home_away_ratio"]
    X = df[feature_cols].to_numpy()
    y = df["label"].astype(int).to_numpy()

    print(f"[train] X shape={X.shape}, y shape={y.shape}")
    print(f"[train] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    params = dict(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        n_jobs=-1,
        random_state=42,
    )

    # ------------------------------
    # Cross-validation
    # ------------------------------
    n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    briers, loglosses, aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Xval, ytr, yval = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params, n_estimators=300)

        # ✅ Use callbacks instead of early_stopping_rounds
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )

        p_raw = model.predict_proba(Xval)[:, 1]
        brier = brier_score_loss(yval, p_raw)
        ll = log_loss(yval, p_raw)
        auc = roc_auc_score(yval, p_raw)
        briers.append(brier)
        loglosses.append(ll)
        aucs.append(auc)
        print(f"  fold {fold}: Brier={brier:.4f}, LogLoss={ll:.4f}, AUC={auc:.4f}")

    print(f"[train] Mean metrics → Brier={np.mean(briers):.4f}, LogLoss={np.mean(loglosses):.4f}, AUC={np.mean(aucs):.4f}")

    # ------------------------------
    # Final training on all data
    # ------------------------------
    print("[train] Training final LightGBMClassifier on all data...")
    model = lgb.LGBMClassifier(**params, n_estimators=300)
    model.fit(X, y)

    print("[train] Calibrating probability outputs...")
    cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
    cal.fit(X, y)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(cal, out_dir / "calibrator.joblib")
    print(f"[train] ✅ Saved model + calibrator → {out_dir}")

    # ------------------------------
    # MLflow logging
    # ------------------------------
    mlflow.set_experiment(args.league)
    with mlflow.start_run(run_name=f"{args.league}_train"):
        mlflow.log_params(params)
        mlflow.log_metrics(
            {"brier_mean": np.mean(briers), "logloss_mean": np.mean(loglosses), "auc_mean": np.mean(aucs)}
        )
        mlflow.log_artifact(out_dir / "model.joblib")
        mlflow.log_artifact(out_dir / "calibrator.joblib")
        print("[train] ✅ MLflow logging complete")


if __name__ == "__main__":
    main()
