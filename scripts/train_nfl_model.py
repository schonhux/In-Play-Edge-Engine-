import polars as pl
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

def main():
    print("[train_nfl_model] 🏈 Training simple NFL model...")

    # Load cleaned data
    df = pl.read_parquet("data/warehouse/NFL/current_team_stats.parquet")

    # Features & target
    X = df.select(["NET_YPP", "PLUS_MINUS", "YPP"]).to_numpy()
    y = df["WIN"].to_numpy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Base model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Calibrator
    cal = CalibratedClassifierCV(model, cv="prefit")
    cal.fit(X_test, y_test)

    # ✅ Save both models correctly
    joblib.dump(model, "artifacts/NFL/model.joblib")
    joblib.dump(cal, "artifacts/NFL/calibrator.joblib")

    print("✅ Saved model → artifacts/NFL/model.joblib")
    print("✅ Saved calibrator → artifacts/NFL/calibrator.joblib")

if __name__ == "__main__":
    main()
