import pandas as pd
import joblib
import shap


def generate_feature_importance():
    df = pd.read_csv("data/processed/train.csv")

    X = df.drop(columns=[
        "datetime",
        "target_pm25_48h",
        "spike_48h"
    ])

    model = joblib.load("models/xgboost_spike_classifier.pkl")

    # On limite l'échantillon pour éviter que SHAP soit trop lent
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "importance": abs(shap_values).mean(axis=0)
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    importance.to_csv(
        "data/processed/shap_feature_importance.csv",
        index=False
    )

    print("✅ SHAP feature importance saved")
    print(importance.head(15))


if __name__ == "__main__":
    generate_feature_importance()