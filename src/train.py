import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# target column -> column with the current value (used by the naive baseline)
TARGETS = {
    "target_us_aqi_24h": "us_aqi",
    "target_temperature_2m_24h": "temperature_2m",
    "target_relative_humidity_2m_24h": "relative_humidity_2m",
    "target_wind_speed_10m_24h": "wind_speed_10m",
    "target_uv_index_24h": "uv_index",
    "target_precipitation_24h": "precipitation",
    "target_wind_dir_sin_24h": "wind_dir_sin",
    "target_wind_dir_cos_24h": "wind_dir_cos",
}
RAIN = "target_rain_next24h"


def split(df):
    n = len(df)
    a, b = int(n * 0.70), int(n * 0.85)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def reg_scores(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p))}


def clf_scores(y, pred, score):
    auc = roc_auc_score(y, score) if y.nunique() == 2 else np.nan
    return {"F1": float(f1_score(y, pred, zero_division=0)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "ROC_AUC": float(auc)}


def main():
    df = pd.read_csv("data/features.csv", parse_dates=["time"], index_col="time").dropna()
    feats = [c for c in df.columns if not c.startswith("target_") and c != "was_imputed"]
    train, val, test = split(df)
    parts = [("validation", val), ("test", test)]

    print(f"Features: {len(feats)}")
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        print(f"{name:10s} {len(part):6d} rows   {part.index.min().date()} -> {part.index.max().date()}")

    models = {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.5,
            n_jobs=-1, random_state=42),
    }

    # regression metrics
    rows = []
    for target, current in TARGETS.items():
        print("training", target)
        for split_name, part in parts:
            rows.append({"target": target, "model": "naive_24h", "split": split_name,
                         **reg_scores(part[target], part[current])})
        for name, make in models.items():
            m = make().fit(train[feats], train[target])
            for split_name, part in parts:
                rows.append({"target": target, "model": name, "split": split_name,
                             **reg_scores(part[target], m.predict(part[feats]))})
    res = pd.DataFrame(rows)

    # rain (yes/no in the next 24h)
    ytrain = train[RAIN].astype(int)
    clfs = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42),
    }
    rain_rows = []
    for split_name, part in parts:
        y = part[RAIN].astype(int)
        base = (part["precipitation_roll24h_mean"] > 0).astype(int)  # any rain in the last 24h
        rain_rows.append({"model": "naive_24h", "split": split_name, **clf_scores(y, base, base)})
    for name, clf in clfs.items():
        clf.fit(train[feats], ytrain)
        for split_name, part in parts:
            y = part[RAIN].astype(int)
            prob = clf.predict_proba(part[feats])[:, 1]
            rain_rows.append({"model": name, "split": split_name,
                              **clf_scores(y, (prob >= 0.5).astype(int), prob)})
    rain = pd.DataFrame(rain_rows)

    res.to_csv("data/results_sprint2.csv", index=False)
    rain.to_csv("data/results_rain_sprint2.csv", index=False)

    print("\n===== TEST SET RESULTS =====")
    for t, g in res[res.split == "test"].round(3).groupby("target", sort=False):
        print("\n" + t)
        print(g[["model", "RMSE", "MAE", "R2"]].to_string(index=False))

    print("\n===== BEST MODEL PER TARGET (chosen on validation RMSE) =====")
    for t in TARGETS:
        v = res[(res.target == t) & (res.split == "validation") & (res.model != "naive_24h")]
        best = v.loc[v.RMSE.idxmin(), "model"]
        te = res[(res.target == t) & (res.split == "test")].set_index("model")
        gain = (1 - te.loc[best, "RMSE"] / te.loc["naive_24h", "RMSE"]) * 100
        print(f"{t:34s} {best:13s} test R2 {te.loc[best, 'R2']:.3f}  "
              f"naive {te.loc['naive_24h', 'R2']:.3f}  (RMSE {gain:+.1f}% vs naive)")

    print("\n===== RAIN IN NEXT 24H (test set) =====")
    print(f"share of rainy hours in test: {test[RAIN].mean() * 100:.2f}%")
    print(rain[rain.split == "test"].round(3)[["model", "F1", "Precision", "Recall", "ROC_AUC"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()