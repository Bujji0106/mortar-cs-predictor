# retrain_and_validate_from_pdf.py
# Retrains a continuous-GO model from the experimental CSV extracted from your PDF,
# evaluates it, saves model + plots, and writes a small validation CSV.

import os, pickle, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# --- configuration ---
BASE = r"C:\Users\bujji\OneDrive\Desktop\final_outputs_no_pdf"
Path(BASE).mkdir(parents=True, exist_ok=True)

INPUT_CSV = os.path.join(BASE, "experimental_dataset_used_from_pdf.csv")
OUT_DIR = os.path.join(BASE, "retrain_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_OUT = os.path.join(OUT_DIR, "gb_model_retrained_from_pdf.pkl")
PERF_CSV = os.path.join(OUT_DIR, "retrain_performance.csv")
PARITY_PNG = os.path.join(OUT_DIR, "parity_retrained_pdf.png")
BAR_DIR = os.path.join(OUT_DIR, "bar_charts")
os.makedirs(BAR_DIR, exist_ok=True)
VALIDATION_CSV = os.path.join(OUT_DIR, "validation_table_model_vs_experiment.csv")

# --- load dataset prepared from PDF (should exist) ---
if not os.path.exists(INPUT_CSV):
    raise SystemExit(f"Input CSV not found: {INPUT_CSV}\nMake sure experimental_dataset_used_from_pdf.csv is present in {BASE}")

df = pd.read_csv(INPUT_CSV)
# Ensure columns used are present
expected_cols = {"GO_frac","GO_pct","environment","day","weight_change_pct","compressive_strength_MPa"}
if not expected_cols.issubset(set(df.columns)):
    # if GO_frac missing but GO_pct present, compute
    if "GO_pct" in df.columns and "GO_frac" not in df.columns:
        df["GO_frac"] = df["GO_pct"] / 100.0
    else:
        raise SystemExit("CSV missing required columns: " + ", ".join(expected_cols))

# create polynomial features the model expects
df["GO_frac_sq"] = df["GO_frac"] ** 2
df["GO_frac_cu"] = df["GO_frac"] ** 3

# features + target
X = df[["GO_frac","GO_frac_sq","GO_frac_cu","environment","day","weight_change_pct"]]
y = df["compressive_strength_MPa"]

# pipeline (one-hot env, pass thru numeric)
pre = ColumnTransformer([("env", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["environment"])], remainder="passthrough")
pipeline = make_pipeline(pre, GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42))

# cross-validated metrics (5-fold)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
cv_rmse = np.sqrt(cv_mse)
cv_r2 = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1)

# train final model on all data
pipeline.fit(X, y)

# save model
with open(MODEL_OUT, "wb") as f:
    pickle.dump(pipeline, f)

# Evaluate predictions on dataset
y_pred = pipeline.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y, y_pred)

perf = {
    "cv_rmse_mean": float(np.mean(cv_rmse)),
    "cv_rmse_std": float(np.std(cv_rmse)),
    "cv_r2_mean": float(np.mean(cv_r2)),
    "train_rmse": float(rmse),
    "train_r2": float(r2),
    "n_samples": int(len(df))
}
pd.DataFrame([perf]).to_csv(PERF_CSV, index=False)

# Parity plot
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.8)
lims = [min(y.min(), y_pred.min())-1, max(y.max(), y_pred.max())+1]
plt.plot(lims, lims, color='black', linewidth=1)
plt.xlabel("Actual CS (MPa)")
plt.ylabel("Predicted CS (MPa)")
plt.title("Parity Plot - Retrained on PDF data")
plt.xlim(lims); plt.ylim(lims)
plt.savefig(PARITY_PNG, bbox_inches="tight")
plt.close()

# Create model vs experiment validation table at 7/28/56 for each environment and GO%
records = []
for env in sorted(df["environment"].unique()):
    for go in sorted(df["GO_pct"].unique()):
        for day in sorted(df["day"].unique()):
            subset = df[(df["environment"]==env) & (df["GO_pct"]==go) & (df["day"]==day)]
            if subset.shape[0] == 0:
                continue
            exp_mean = float(subset["compressive_strength_MPa"].mean())
            exp_std = float(subset["compressive_strength_MPa"].std(ddof=0)) if subset.shape[0]>1 else 0.0
            # create single-row X to predict
            rowX = {
                "GO_frac": go/100.0,
                "GO_frac_sq": (go/100.0)**2,
                "GO_frac_cu": (go/100.0)**3,
                "environment": env,
                "day": day,
                "weight_change_pct": float(subset["weight_change_pct"].mean())
            }
            Xpred = pd.DataFrame([rowX])
            try:
                ymodel = float(pipeline.predict(Xpred)[0])
            except Exception as e:
                ymodel = float('nan')
            records.append({
                "environment": env,
                "GO_pct": go,
                "day": day,
                "exp_mean_MPa": exp_mean,
                "exp_std_MPa": exp_std,
                "model_pred_MPa": ymodel,
                "abs_err": abs(ymodel-exp_mean) if not np.isnan(ymodel) else None,
                "rel_err_pct": 100.0*abs(ymodel-exp_mean)/exp_mean if (exp_mean!=0 and not np.isnan(ymodel)) else None
            })

val_df = pd.DataFrame(records)
val_df.to_csv(VALIDATION_CSV, index=False)

# Bar charts per environment comparing experiment vs model predictions across GO% for each day (7,28,56)
days_show = sorted(df["day"].unique())
bar_files = []
for env in sorted(df["environment"].unique()):
    fig, axs = plt.subplots(1, len(days_show), figsize=(4*len(days_show),4), sharey=True)
    for i, d in enumerate(days_show):
        sub = val_df[(val_df["environment"]==env) & (val_df["day"]==d)]
        gos = sub["GO_pct"].values
        exp_vals = sub["exp_mean_MPa"].values
        model_vals = sub["model_pred_MPa"].values
        x = np.arange(len(gos))
        w = 0.35
        axs[i].bar(x - w/2, exp_vals, w, label="Experiment")
        axs[i].bar(x + w/2, model_vals, w, label="Model")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels([f"{g:.2f}" for g in gos])
        axs[i].set_title(f"{d} days")
        axs[i].set_xlabel("GO %")
        if i==0:
            axs[i].set_ylabel("Compressive Strength (MPa)")
    axs[-1].legend()
    fig.suptitle(f"Model vs Experiment — {env}")
    outp = os.path.join(BAR_DIR, f"bar_compare_{env}.png")
    fig.savefig(outp, bbox_inches="tight")
    bar_files.append(outp)
    plt.close(fig)

# Write a small summary text
summary_path = os.path.join(OUT_DIR, "retrain_summary.txt")
with open(summary_path, "w") as f:
    f.write("Retrain summary\n")
    f.write(str(perf) + "\n")
    f.write("Parity plot: " + PARITY_PNG + "\n")
    f.write("Bar charts: " + ", ".join(bar_files) + "\n")
    f.write("Validation CSV: " + VALIDATION_CSV + "\n")

print("Retrain complete.")
print("Model saved to:", MODEL_OUT)
print("Performance CSV:", PERF_CSV)
print("Parity plot:", PARITY_PNG)
print("Validation CSV:", VALIDATION_CSV)
print("Bar charts saved in:", BAR_DIR)
