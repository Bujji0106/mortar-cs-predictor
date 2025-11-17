# retrain_tune_exposure.py
import os, pickle, math, time
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
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
# Ensure required columns
if "GO_frac" not in df.columns and "GO_pct" in df.columns:
    df["GO_frac"] = df["GO_pct"] / 100.0
if "GO_frac_sq" not in df.columns:
    df["GO_frac_sq"] = df["GO_frac"] ** 2
if "GO_frac_cu" not in df.columns:
    df["GO_frac_cu"] = df["GO_frac"] ** 3

# base engineered features
df["day_sq"] = df["day"] ** 2
df["day_log"] = np.log1p(df["day"])
df["GOxday"] = df["GO_frac"] * df["day"]
df["GOxwt"] = df["GO_frac"] * df["weight_change_pct"]

# Build cum_wt_loss via interpolation per (env,GO), normalized per-group
def cum_weight_for_sub(sub):
    x = sub["day"].values
    y = sub["weight_change_pct"].values
    order = np.argsort(x)
    x,y = x[order], y[order]
    if len(x)==1:
        days = np.arange(1,366)
        series = np.full(len(days), float(y[0]))
        cum = np.cumsum(series)
        return days, series, cum
    f = PchipInterpolator(x, y, extrapolate=True)
    days = np.arange(1,366)
    series = f(days)
    series = np.maximum(series, 0.0)
    cum = np.cumsum(series)
    return days, series, cum

cum_maps = {}
for (env, go), sub in df.groupby(["environment","GO_pct"]):
    days_arr, wt_series, cum_series = cum_weight_for_sub(sub)
    cum_maps[(env, round(go,2))] = {"days": days_arr, "wt": wt_series, "cum": cum_series}

df["cum_wt_loss"] = 0.0
for idx,row in df.iterrows():
    key=(row["environment"], round(row["GO_pct"],2))
    if key in cum_maps:
        d=int(row["day"])
        df.at[idx,"cum_wt_loss"]=float(cum_maps[key]["cum"][d-1])
    else:
        df.at[idx,"cum_wt_loss"]=0.0

# normalize cum_wt_loss per group to avoid scale issues
df["cum_wt_norm"] = 0.0
for (env,go), sub in df.groupby(["environment","GO_pct"]):
    key=(env,round(go,2))
    cum = cum_maps.get(key, None)
    if cum is not None:
        maxcum = float(np.max(cum["cum"])) if len(cum["cum"])>0 else 1.0
        idxs = sub.index
        for i in idxs:
            d = int(df.at[i,"day"])
            df.at[i,"cum_wt_norm"] = df.at[i,"cum_wt_loss"] / (maxcum if maxcum>0 else 1.0)
    else:
        df.loc[sub.index,"cum_wt_norm"] = 0.0

# more features
df["sqrt_day"] = np.sqrt(df["day"])
df["cum_wt_per_day"] = df["cum_wt_loss"] / df["day"].replace(0,1)

# final feature set
feature_cols = ["GO_frac","GO_frac_sq","GO_frac_cu","environment","day","day_sq","day_log",
                "sqrt_day","GOxday","GOxwt","weight_change_pct","cum_wt_loss","cum_wt_norm","cum_wt_per_day"]

# fill any nans
df[feature_cols] = df[feature_cols].fillna(0.0)

X = df[feature_cols]
y = df["compressive_strength_MPa"].values

# groups to use GroupKFold — group by environment+GO_pct so same group doesn't spread across folds
groups = df["environment"].astype(str) + "_" + df["GO_pct"].astype(str)

# pipeline: OneHot environment, pass-through numeric (no scaling for trees)
pre = ColumnTransformer([("env", OneHotEncoder(handle_unknown="ignore", sparse=False), ["environment"])], remainder="passthrough")

est = GradientBoostingRegressor(random_state=42)

pipe = make_pipeline(pre, est)

# parameter distribution for RandomizedSearchCV
param_dist = {
    "gradientboostingregressor__n_estimators": [100,200,300,400,600],
    "gradientboostingregressor__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "gradientboostingregressor__max_depth": [2,3,4,5,6],
    "gradientboostingregressor__subsample": [0.6,0.7,0.8,1.0],
    "gradientboostingregressor__min_samples_leaf": [1,2,3,5,10],
    "gradientboostingregressor__max_features": [None, "sqrt", 0.5]
}

cv = GroupKFold(n_splits=5)

search = RandomizedSearchCV(pipe, param_dist, n_iter=40, scoring="neg_mean_squared_error", cv=cv.split(X,y,groups=groups), verbose=2, n_jobs=-1, random_state=42, return_train_score=True)

start=time.time()
search.fit(X, y)
end=time.time()
print("Search done in %.1f seconds" % (end-start))
best = search.best_estimator_
best_params = search.best_params_
best_score = search.best_score_  # negative MSE
best_rmse = math.sqrt(-best_score)
print("Best RMSE (CV):", best_rmse)
print("Best params:", best_params)

# Save the best model
out_model = os.path.join(OUT_DIR, "gb_model_exposure_tuned.pkl")
with open(out_model, "wb") as f:
    pickle.dump(best, f)

# Evaluate on training data (for diagnostics)
y_pred_train = best.predict(X)
train_rmse = math.sqrt(mean_squared_error(y, y_pred_train))
train_r2 = r2_score(y, y_pred_train)

pd.DataFrame([{"best_cv_rmse":best_rmse,"train_rmse":train_rmse,"train_r2":train_r2,"n_samples":len(df)}]).to_csv(os.path.join(OUT_DIR,"retrain_exposure_tuned_perf.csv"), index=False)

# Build validation CSV like before (use compute-row-features approach)
records=[]
for env in sorted(df["environment"].unique()):
    for go in sorted(df["GO_pct"].unique()):
        for d in sorted(df["day"].unique()):
            sub = df[(df["environment"]==env)&(df["GO_pct"]==go)&(df["day"]==d)]
            if sub.empty: continue
            exp_mean = float(sub["compressive_strength_MPa"].mean())
            exp_std = float(sub["compressive_strength_MPa"].std(ddof=0)) if sub.shape[0]>1 else 0.0
            # build features for this single row
            row = {
                "GO_frac": go/100.0,
                "GO_frac_sq": (go/100.0)**2,
                "GO_frac_cu": (go/100.0)**3,
                "environment": env,
                "day": int(d),
                "day_sq": d**2,
                "day_log": np.log1p(d),
                "sqrt_day": np.sqrt(d),
                "GOxday": (go/100.0)*d,
                "GOxwt": (go/100.0)*float(sub["weight_change_pct"].mean()),
                "weight_change_pct": float(sub["weight_change_pct"].mean()),
                "cum_wt_loss": float(sub["cum_wt_loss"].mean()),
                "cum_wt_norm": float(sub["cum_wt_norm"].mean()),
                "cum_wt_per_day": float(sub["cum_wt_per_day"].mean())
            }
            Xp = pd.DataFrame([row])
            try:
                pred = best.predict(Xp)[0]
            except Exception:
                pred = float('nan')
            abs_err = abs(pred - exp_mean) if not np.isnan(pred) else None
            rel_err = 100.0*abs_err/exp_mean if (exp_mean!=0 and abs_err is not None) else None
            records.append({"environment":env,"GO_pct":go,"day":d,"exp_mean_MPa":exp_mean,"exp_std_MPa":exp_std,"model_pred_MPa":pred,"abs_err":abs_err,"rel_err_pct":rel_err})
valdf = pd.DataFrame(records)
valdf.to_csv(os.path.join(OUT_DIR,"validation_table_model_vs_experiment_exposure_tuned.csv"), index=False)

print("Tuning complete. Best CV RMSE:", best_rmse)
print("Model saved to:", out_model)
print("Validation CSV saved to:", os.path.join(OUT_DIR,"validation_table_model_vs_experiment_exposure_tuned.csv"))
