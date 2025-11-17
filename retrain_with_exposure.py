# retrain_with_exposure.py  (fixed)
import os, pickle, math
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import LinearRegression

BASE = r"C:\Users\bujji\OneDrive\Desktop\final_outputs_no_pdf"
IN_CSV = os.path.join(BASE, "experimental_dataset_used_from_pdf.csv")
OUT_DIR = os.path.join(BASE, "retrain_outputs")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

if not os.path.exists(IN_CSV):
    raise SystemExit("Input CSV missing: " + IN_CSV)

df = pd.read_csv(IN_CSV)

# Ensure GO_frac exists
if "GO_frac" not in df.columns and "GO_pct" in df.columns:
    df["GO_frac"] = df["GO_pct"] / 100.0
if "GO_frac_sq" not in df.columns:
    df["GO_frac_sq"] = df["GO_frac"] ** 2
if "GO_frac_cu" not in df.columns:
    df["GO_frac_cu"] = df["GO_frac"] ** 3

# Engineered features
df["day_sq"] = df["day"] ** 2
df["day_cu"] = df["day"] ** 3
df["GOxday"] = df["GO_frac"] * df["day"]
df["GOxwt"] = df["GO_frac"] * df["weight_change_pct"]

# Helper: compute an interpolated weight series and cumulative mapping for a given (env, GO)
def cum_weight_for_sub(sub):
    x = sub["day"].values
    y = sub["weight_change_pct"].values
    # sort unique
    order = np.argsort(x)
    x, y = x[order], y[order]
    # if only single point, return that constant series
    if len(x) == 1:
        days = np.arange(1, 366)
        series = np.full(len(days), float(y[0]))
        cum = np.cumsum(series)
        return days, series, cum
    # use PCHIP for smooth monotonic-ish interpolation
    f = PchipInterpolator(x, y, extrapolate=True)
    days = np.arange(1, 366)
    series = f(days)
    series = np.maximum(series, 0.0)
    cum = np.cumsum(series)  # simple cumulative proxy
    return days, series, cum

# Precompute cumulative maps for each (env, GO_pct) and store in dict for quick lookup
cum_maps = {}
for (env, go), sub in df.groupby(["environment", "GO_pct"]):
    days_arr, wt_series, cum_series = cum_weight_for_sub(sub)
    cum_maps[(env, round(go,2))] = {
        "days": days_arr,
        "wt_series": wt_series,
        "cum_series": cum_series
    }

# Add cum_wt_loss column to measured rows (value of cumulative proxy up to measured day)
df["cum_wt_loss"] = 0.0
for idx, row in df.iterrows():
    key = (row["environment"], round(row["GO_pct"],2))
    if key in cum_maps:
        d = int(row["day"])
        cum_val = cum_maps[key]["cum_series"][d-1]  # d in 1..365
        df.at[idx, "cum_wt_loss"] = float(cum_val)
    else:
        df.at[idx, "cum_wt_loss"] = 0.0

# Final feature set
feature_cols = ["GO_frac", "GO_frac_sq", "GO_frac_cu", "environment",
                "day", "day_sq", "GOxday", "GOxwt", "cum_wt_loss", "weight_change_pct"]
X = df[feature_cols]
y = df["compressive_strength_MPa"]

# Pipeline & training
pre = ColumnTransformer([("env", OneHotEncoder(handle_unknown="ignore"), ["environment"])], remainder="passthrough")
pipeline = make_pipeline(pre, GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42))

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
cv_rmse = np.sqrt(cv_scores)
print("CV RMSE mean:", cv_rmse.mean(), "std:", cv_rmse.std())

pipeline.fit(X, y)

# save model
out_model = os.path.join(OUT_DIR, "gb_model_exposure_trained.pkl")
with open(out_model, "wb") as f:
    pickle.dump(pipeline, f)

# Helper: build the exact feature dict for a given env, go_pct, day, weight_change_pct
def compute_row_features_for_model(env_chosen, go_pct_chosen, day_chosen, weight_change_pct_val):
    go_frac = float(go_pct_chosen) / 100.0
    go_sq = go_frac ** 2
    go_cu = go_frac ** 3
    day_sq = int(day_chosen) ** 2
    goxday = go_frac * int(day_chosen)
    goxwt = go_frac * float(weight_change_pct_val)
    # cumulative wt loss: if mapping exists use it; else compute from env-level mean
    key = (env_chosen, round(go_pct_chosen,2))
    if key in cum_maps:
        cum_val = float(cum_maps[key]["cum_series"][int(day_chosen)-1])
    else:
        # fallback: take environment-level mean series if available
        sub = df[df["environment"]==env_chosen].groupby("day")["weight_change_pct"].mean().reset_index()
        if sub.shape[0] >= 1:
            xx = sub["day"].values
            yy = sub["weight_change_pct"].values
            order = np.argsort(xx)
            xx, yy = xx[order], yy[order]
            if len(xx) == 1:
                series = np.full(365, float(yy[0]))
            else:
                f = PchipInterpolator(xx, yy, extrapolate=True)
                series = f(np.arange(1,366))
                series = np.maximum(series, 0.0)
            cum_val = float(np.cumsum(series)[int(day_chosen)-1])
        else:
            cum_val = 0.0
    return {
        "GO_frac": go_frac,
        "GO_frac_sq": go_sq,
        "GO_frac_cu": go_cu,
        "environment": env_chosen,
        "day": int(day_chosen),
        "day_sq": day_sq,
        "GOxday": goxday,
        "GOxwt": goxwt,
        "cum_wt_loss": cum_val,
        "weight_change_pct": float(weight_change_pct_val)
    }

# Build validation CSV (use model predictions; compute features properly)
records = []
for env in sorted(df["environment"].unique()):
    for go in sorted(df["GO_pct"].unique()):
        for d in sorted(df["day"].unique()):
            sub = df[(df["environment"]==env) & (df["GO_pct"]==go) & (df["day"]==d)]
            if sub.empty:
                continue
            exp_mean = float(sub["compressive_strength_MPa"].mean())
            exp_std = float(sub["compressive_strength_MPa"].std(ddof=0)) if sub.shape[0] > 1 else 0.0
            wt_val = float(sub["weight_change_pct"].mean())
            feat = compute_row_features_for_model(env, go, d, wt_val)
            Xp = pd.DataFrame([feat])
            try:
                model_pred = float(pipeline.predict(Xp)[0])
            except Exception as e:
                model_pred = float("nan")
            abs_err = abs(model_pred - exp_mean) if not np.isnan(model_pred) else None
            rel_err = 100.0 * abs_err / exp_mean if (exp_mean != 0 and abs_err is not None) else None
            records.append({
                "environment": env,
                "GO_pct": go,
                "day": int(d),
                "exp_mean_MPa": exp_mean,
                "exp_std_MPa": exp_std,
                "weight_change_pct": wt_val,
                "model_pred_MPa": model_pred,
                "abs_err": abs_err,
                "rel_err_pct": rel_err
            })

val_df = pd.DataFrame(records)
val_csv_path = os.path.join(OUT_DIR, "validation_table_model_vs_experiment_exposure.csv")
val_df.to_csv(val_csv_path, index=False)

# Save performance summary
y_pred = pipeline.predict(X)
mse = mean_squared_error(y, y_pred); rmse = math.sqrt(mse); r2 = r2_score(y, y_pred)
perf = {"cv_rmse_mean": float(cv_rmse.mean()), "cv_rmse_std": float(cv_rmse.std()),
        "train_rmse": float(rmse), "train_r2": float(r2), "n_samples": int(len(df))}
pd.DataFrame([perf]).to_csv(os.path.join(OUT_DIR, "retrain_exposure_perf.csv"), index=False)

print("Retrain with exposure features complete.")
print("Model saved to:", out_model)
print("Validation CSV saved to:", val_csv_path)
print("Performance summary saved to:", os.path.join(OUT_DIR, "retrain_exposure_perf.csv"))
