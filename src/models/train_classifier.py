import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def load_data(path="data/processed/train.csv"):
    return pd.read_csv(path)


def train_classifier(df):

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train.drop(
        columns=["datetime", "target_pm25_48h", "spike_48h"]
    )

    y_train = train["spike_48h"]

    X_test = test.drop(
        columns=["datetime", "target_pm25_48h", "spike_48h"]
    )

    y_test = test["spike_48h"]

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    print("scale_pos_weight:", scale_pos_weight)

    model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)


    probs = model.predict_proba(X_test)[:, 1]

    threshold = 0.30
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/xgboost_spike_classifier.pkl")

    print("✅ Classifier sauvegardé")


if __name__ == "__main__":
    df = load_data()
    train_classifier(df)