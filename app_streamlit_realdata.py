# app_streamlit_realdata.py
# Ultra-Luxe Research Dashboard — tuned exposure model loaded by default (gb_model_exposure_tuned.pkl)
# This app computes predictions using the SAME engineered features used during retraining,
# builds day-1..365 time-series predictions from the model, applies optional depletion mapping,
# and shows publication-ready visuals + validation downloads.

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, io
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Configuration - change if your folder is different
# ---------------------------
BASE = r"C:\Users\bujji\OneDrive\Desktop\final_outputs_no_pdf"
# tuned exposure-trained model produced by retrain_tune_exposure.py
TUNED_MODEL = os.path.join(BASE, "retrain_outputs", "gb_model_exposure_tuned.pkl")
# fallback models
EXPOSURE_MODEL_FALLBACK = os.path.join(BASE, "retrain_outputs", "gb_model_exposure_trained.pkl")
RETRAIN_MODEL_OLD = os.path.join(BASE, "retrain_outputs", "gb_model_retrained_from_pdf.pkl")
DEFAULT_MODEL = os.path.join(BASE, "gb_model_continuous.pkl")

EXPCSV = os.path.join(BASE, "experimental_dataset_used_from_pdf.csv")
VALIDATION_TUNED = os.path.join(BASE, "retrain_outputs", "validation_table_model_vs_experiment_exposure_tuned.csv")
VALIDATION_BASE = os.path.join(BASE, "retrain_outputs", "validation_table_model_vs_experiment.csv")
PARITY_IMG = os.path.join(BASE, "retrain_outputs", "parity_retrained_pdf.png")
PERF_TUNED = os.path.join(BASE, "retrain_outputs", "retrain_exposure_tuned_perf.csv")
PERF_BASE = os.path.join(BASE, "retrain_outputs", "retrain_exposure_perf.csv")
JOURNALS_TXT = os.path.join(BASE, "validation_plan_fulltitles.txt")
BAR_DIR = os.path.join(BASE, "retrain_outputs", "bar_charts")

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Mortar CS Predictor — Tuned Exposure Model", layout="wide")
st.title("Mortar Compressive Strength Predictor — Tuned Exposure Model (Ultra-Luxe)")

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data
def load_model_prefer_tuned():
    # prefer tuned exposure-trained model; fallbacks after
    for p in [TUNED_MODEL, EXPOSURE_MODEL_FALLBACK, RETRAIN_MODEL_OLD, DEFAULT_MODEL]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f), p
            except Exception:
                continue
    return None, None

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def ensure_features(df):
    df = df.copy()
    if "GO_frac" not in df.columns and "GO_pct" in df.columns:
        df["GO_frac"] = df["GO_pct"] / 100.0
    if "GO_pct" not in df.columns and "GO_frac" in df.columns:
        df["GO_pct"] = df["GO_frac"] * 100.0
    if "GO_frac_sq" not in df.columns and "GO_frac" in df.columns:
        df["GO_frac_sq"] = df["GO_frac"]**2
    if "GO_frac_cu" not in df.columns and "GO_frac" in df.columns:
        df["GO_frac_cu"] = df["GO_frac"]**3
    return df

def safe_errbars(std_series):
    if std_series.isna().all():
        return None
    arr = np.nan_to_num(std_series.values, nan=0.0)
    if np.all(arr == 0.0):
        return None
    return arr

# interpolation / cumulative helpers
@st.cache_data
def build_cum_maps(experiment_df):
    cum_maps = {}
    for (env, go), sub in experiment_df.groupby(["environment", "GO_pct"]):
        x = sub["day"].values
        y = sub["weight_change_pct"].values
        order = np.argsort(x)
        x, y = x[order], y[order]
        if len(x) == 1:
            days = np.arange(1,366)
            series = np.full(len(days), float(y[0]))
            cum = np.cumsum(series)
        else:
            f = PchipInterpolator(x, y, extrapolate=True)
            days = np.arange(1,366)
            series = f(days)
            series = np.maximum(series, 0.0)
            cum = np.cumsum(series)
        cum_maps[(env, round(go,2))] = {"days": days, "wt": series, "cum": cum}
    return cum_maps

def weight_series_for(env_chosen, go_chosen, cum_maps):
    key = (env_chosen, round(go_chosen,2))
    if key in cum_maps:
        return cum_maps[key]["days"], cum_maps[key]["wt"], cum_maps[key]["cum"]
    # fallback: environment-level average
    return None, None, None

def compute_row_features(env_chosen, go_pct_chosen, day_chosen, weight_change_pct_val, cum_maps, experiment_df):
    go_frac = float(go_pct_chosen)/100.0
    feat = {}
    feat["GO_frac"] = go_frac
    feat["GO_frac_sq"] = go_frac**2
    feat["GO_frac_cu"] = go_frac**3
    feat["environment"] = env_chosen
    feat["day"] = int(day_chosen)
    feat["day_sq"] = int(day_chosen)**2
    feat["day_log"] = np.log1p(int(day_chosen))
    feat["sqrt_day"] = np.sqrt(int(day_chosen))
    feat["GOxday"] = go_frac * int(day_chosen)
    feat["GOxwt"] = go_frac * float(weight_change_pct_val)
    # cum wt: use cum_maps if exists else fallback to env-average interpolation
    key = (env_chosen, round(go_pct_chosen,2))
    if key in cum_maps:
        cum_val = float(cum_maps[key]["cum"][int(day_chosen)-1])
    else:
        # fallback - build local interpolation for env
        sub = experiment_df[experiment_df["environment"]==env_chosen].groupby("day")["weight_change_pct"].mean().reset_index()
        if sub.shape[0] == 0:
            cum_val = 0.0
        else:
            xx = sub["day"].values; yy = sub["weight_change_pct"].values
            order = np.argsort(xx)
            xx, yy = xx[order], yy[order]
            if len(xx) == 1:
                series = np.full(365, float(yy[0]))
            else:
                f = PchipInterpolator(xx, yy, extrapolate=True)
                series = f(np.arange(1,366))
                series = np.maximum(series, 0.0)
            cum_val = float(np.cumsum(series)[int(day_chosen)-1])
    feat["cum_wt_loss"] = cum_val
    feat["weight_change_pct"] = float(weight_change_pct_val)
    # also keep cum normalized and cum per day (if model was trained with these, safe to include; model will ignore unknown columns)
    # normalized cum: divide by max cum for group if present
    maxcum = 1.0
    key2 = (env_chosen, round(go_pct_chosen,2))
    if key2 in cum_maps:
        maxcum = float(np.max(cum_maps[key2]["cum"])) if np.max(cum_maps[key2]["cum"])>0 else 1.0
    feat["cum_wt_norm"] = feat["cum_wt_loss"] / maxcum
    feat["cum_wt_per_day"] = feat["cum_wt_loss"]/max(1,int(day_chosen))
    return feat

# ---------------------------
# Load model and data
# ---------------------------
model, model_path = load_model_prefer_tuned()
experiment_df = load_csv(EXPCSV)
validation_tuned_df = load_csv(VALIDATION_TUNED)
validation_base_df = load_csv(VALIDATION_BASE)
perf_tuned_df = load_csv(PERF_TUNED)
perf_base_df = load_csv(PERF_BASE)
journal_titles = []
if os.path.exists(JOURNALS_TXT):
    try:
        with open(JOURNALS_TXT, "r", encoding="utf-8") as f:
            journal_titles = [l.strip() for l in f.readlines() if l.strip()]
    except Exception:
        journal_titles = []

if experiment_df is None:
    st.error(f"Experimental CSV not found at {EXPCSV}. Place experimental_dataset_used_from_pdf.csv in your project folder.")
    st.stop()

experiment_df = ensure_features(experiment_df)
cum_maps = build_cum_maps(experiment_df)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Inputs")
env_options = sorted(experiment_df["environment"].unique().tolist())
env = st.sidebar.selectbox("Environment", env_options)
go_pct = st.sidebar.slider("GO (%) by weight of cement", 0.00, 0.10, 0.06, step=0.001)
pred_day = st.sidebar.number_input("Prediction day (single metric)", min_value=1, max_value=365, value=28)
weight_change_input = st.sidebar.number_input("Manual weight change (%) (optional - overrides interpolation)", value=0.0, format="%.2f")
st.sidebar.markdown("---")
st.sidebar.header("Depletion & visual controls")
enable_depletion = st.sidebar.checkbox("Apply weight-loss → strength-loss depletion mapping (Option B)", value=True)
apply_visual_depletion_if_no_alpha = st.sidebar.checkbox("Apply parabolic visual depletion when alpha not available", value=True)
T_visual = st.sidebar.slider("Visual depletion time T (days)", 30, 365, 90)
p_visual = st.sidebar.slider("Visual parabola power p", 1.0, 5.0, 2.0, step=0.1)
st.sidebar.markdown("---")
st.sidebar.header("App & downloads")
show_journals = st.sidebar.checkbox("Show validating journal titles", True)
st.sidebar.write("Loaded model:")
st.sidebar.write(os.path.basename(model_path) if model_path else "No model found")

# ---------------------------
# Fit alpha mapping (rel_strength_loss = alpha * weight_pct + beta) from validation tuned CSV if available
# ---------------------------
alpha = 0.0; beta = 0.0; alpha_status = "Not available"
if validation_tuned_df is None and validation_base_df is not None:
    validation_tuned_df = validation_base_df
if validation_tuned_df is not None:
    try:
        ctrl = validation_tuned_df[validation_tuned_df["environment"]=="control"][["GO_pct","day","exp_mean_MPa"]].rename(columns={"exp_mean_MPa":"exp_control_MPa"})
        chem = validation_tuned_df[validation_tuned_df["environment"]!="control"].merge(ctrl, on=["GO_pct","day"], how="left")
        chem = chem.dropna(subset=["exp_control_MPa","exp_mean_MPa"])
        if not chem.empty:
            chem["rel_strength_loss"] = 1.0 - (chem["exp_mean_MPa"] / chem["exp_control_MPa"])
            # merge weight change from experiment_df
            wt_df = experiment_df[["environment","GO_pct","day","weight_change_pct"]]
            chem = chem.merge(wt_df, left_on=["environment","GO_pct","day"], right_on=["environment","GO_pct","day"], how="left")
            chem = chem.dropna(subset=["weight_change_pct"])
            if not chem.empty:
                lr = LinearRegression()
                lr.fit(chem[["weight_change_pct"]].values, chem["rel_strength_loss"].values)
                alpha = float(lr.coef_[0]); beta = float(lr.intercept_)
                alpha_status = f"alpha={alpha:.6f}, beta={beta:.6f}"
    except Exception as e:
        alpha_status = f"Alpha fit error: {e}"

# ---------------------------
# Single-point prediction (compute engineered features exactly as training used)
# ---------------------------
st.header("Single-point prediction & quick validation")
colA, colB = st.columns([2,1])
with colA:
    st.subheader("Predict compressive strength")
    feat_single = compute_row_features(env, go_pct, pred_day, weight_change_input if weight_change_input>0 else 0.0, cum_maps, experiment_df)
    X_single = pd.DataFrame([feat_single])
    if model is None:
        st.error("No model loaded. Place a tuned model at retrain_outputs/gb_model_exposure_tuned.pkl or tuned output path.")
        pred_val = None
    else:
        try:
            pred_val = float(model.predict(X_single)[0])
            st.metric("Predicted compressive strength (MPa)", f"{pred_val:.2f}")
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            pred_val = None

with colB:
    st.subheader("Validation summary")
    if validation_tuned_df is not None:
        # summary metrics
        vt = validation_tuned_df.dropna(subset=["exp_mean_MPa","model_pred_MPa"])
        vt["abs_err"] = (vt["model_pred_MPa"] - vt["exp_mean_MPa"]).abs()
        mae = vt["abs_err"].mean()
        rmse = np.sqrt(((vt["model_pred_MPa"] - vt["exp_mean_MPa"])**2).mean())
        st.write(f"Validation rows: {len(vt)}")
        st.write(f"MAE: {mae:.3f} MPa")
        st.write(f"RMSE: {rmse:.3f} MPa")
        st.write("Alpha mapping (rel_strength_loss = alpha * wt% + beta):")
        st.write(alpha_status)
    else:
        st.info("Validation table not found. Run retrain and tuning scripts to generate it.")

# ---------------------------
# Model-based time-series predictions (day 1..365)
# ---------------------------
st.markdown("## Time-series prediction (model-based; day 1 → 365)")

days = np.arange(1,366)
# Build feature rows for days 1..365 (use interpolation of weight_change_pct unless user overrides with a manual value)
_, wt_series_full, cum_full = weight_series_for(env, go_pct, cum_maps)
if wt_series_full is None:
    # fallback: environment-level average interpolation
    # compute from experiment_df
    sub_env = experiment_df[experiment_df["environment"]==env].groupby("day")["weight_change_pct"].mean().reset_index()
    if sub_env.shape[0] > 0:
        x = sub_env["day"].values; y = sub_env["weight_change_pct"].values
        f = PchipInterpolator(x, y, extrapolate=True) if len(x)>1 else None
        if f is None:
            wt_series_full = np.full(len(days), float(y[0]))
        else:
            wt_series_full = np.maximum(f(days), 0.0)
    else:
        wt_series_full = np.zeros(len(days))
    cum_full = np.cumsum(wt_series_full)

# if user provided a manual weight_change_input (>0), override the interpolated series at measured day positions only
if weight_change_input > 0:
    # override all days to constant user value (simple behavior)
    wt_series_full = np.full(len(days), float(weight_change_input))
    cum_full = np.cumsum(wt_series_full)

# Build X_days features identical to training features
X_rows = []
for d, wt_val, cum_val in zip(days, wt_series_full, cum_full):
    row = compute_row_features(env, go_pct, int(d), float(wt_val), cum_maps, experiment_df)
    # include variant features used by tuned pipeline (day_log, sqrt_day, cum_wt_norm etc.)
    # compute cum_wt_norm if possible
    key = (env, round(go_pct,2))
    maxcum = 1.0
    if key in cum_maps:
        maxcum = float(np.max(cum_maps[key]["cum"])) if np.max(cum_maps[key]["cum"])>0 else 1.0
    row["cum_wt_norm"] = row["cum_wt_loss"] / maxcum
    row["cum_wt_per_day"] = row["cum_wt_loss"] / max(1,int(d))
    X_rows.append(row)

X_days_df = pd.DataFrame(X_rows)

# predict baseline from model
if model is not None:
    try:
        preds_days_baseline = model.predict(X_days_df)
    except Exception as e:
        st.error("Time-series prediction failed: " + str(e))
        preds_days_baseline = np.full(len(days), np.nan)
else:
    preds_days_baseline = np.full(len(days), np.nan)

# compute depletion multiplier from alpha mapping if enabled and available
if enable_depletion and (alpha is not None):
    mult = 1.0 - (alpha * wt_series_full + beta)
    mult = np.clip(mult, 0.0, 1.0)
    if not np.any(mult > 0):
        # numeric fallback
        mult = 1.0 - (days / float(max(1, T_visual)))**p_visual
        mult[mult < 0] = 0.0
else:
    # visual fallback
    mult = 1.0 - (days / float(max(1, T_visual)))**p_visual
    mult[mult < 0] = 0.0

preds_days_depleted = preds_days_baseline * mult

# Build control baseline (same GO, environment='control') with its own interpolation (control weight loss ~0)
X_control_rows = []
for d in days:
    rowc = compute_row_features("control", go_pct, int(d), 0.0, cum_maps, experiment_df)
    # cum norm for control
    keyc = ("control", round(go_pct,2))
    maxcum_c = 1.0
    if keyc in cum_maps:
        maxcum_c = float(np.max(cum_maps[keyc]["cum"])) if np.max(cum_maps[keyc]["cum"])>0 else 1.0
    rowc["cum_wt_norm"] = rowc["cum_wt_loss"]/maxcum_c
    rowc["cum_wt_per_day"] = rowc["cum_wt_loss"]/max(1,int(d))
    X_control_rows.append(rowc)
X_control_df = pd.DataFrame(X_control_rows)
if model is not None:
    try:
        preds_control = model.predict(X_control_df)
    except Exception as e:
        preds_control = np.full(len(days), np.nan)
else:
    preds_control = np.full(len(days), np.nan)

# ---------------------------
# Plotting time-series (control vs depleted chemical)
# ---------------------------
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(days, preds_control, label="Control baseline (model)", color="#1b9e77", lw=2)
ax.plot(days, preds_days_depleted, label=f"{env} (model + depletion)", color="#d95f02", lw=2)
# overlay measured points same GO & env
meas_same = experiment_df[(experiment_df["environment"]==env) & (np.isclose(experiment_df["GO_pct"], round(go_pct,2)))]
if not meas_same.empty:
    ax.scatter(meas_same["day"], meas_same["compressive_strength_MPa"], color="black", s=50, label="Measured (same GO%)")
# overlay control experimental means
control_means = experiment_df[experiment_df["environment"]=="control"].groupby("day")["compressive_strength_MPa"].agg(["mean","std"]).reindex([7,28,56])
errs_ctrl = safe_errbars(control_means["std"].fillna(np.nan)) if not control_means.empty else None
if errs_ctrl is None:
    if not control_means.empty:
        ax.plot(control_means.index.values, control_means["mean"].values, 's', color='tab:green', label="Control exp mean (7/28/56)")
else:
    ax.errorbar(control_means.index.values, control_means["mean"].values, yerr=errs_ctrl, fmt='s', color='tab:green', label="Control exp mean (7/28/56)", capsize=4)

ax.set_xlabel("Days"); ax.set_ylabel("Compressive strength (MPa)")
ax.set_title(f"Control vs {env} (GO {go_pct:.3f}%) — Model predictions")
ax.grid(alpha=0.25, linestyle="--"); ax.legend()
st.pyplot(fig)
buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
st.download_button("Download control_vs_depletion_curve (PNG)", buf, file_name=f"control_vs_{env}_GO{go_pct:.3f}.png", mime="image/png")

# ---------------------------
# Publication-style grouped bars (7/28/56) comparing experiment, model depleted, control model
# ---------------------------
st.markdown("## 7/28/56 grouped comparison (Experiment vs Model depleted vs Control model)")
days_show = [7,28,56]
# compute predictions for these days
preds_bar_chem_raw = []
preds_bar_ctrl = []
wt_at_days = []
for d in days_show:
    feat = compute_row_features(env, go_pct, int(d), wt_series_full[d-1], cum_maps, experiment_df)
    feat_ctrl = compute_row_features("control", go_pct, int(d), 0.0, cum_maps, experiment_df)
    try:
        pchem_raw = model.predict(pd.DataFrame([feat]))[0] if model is not None else np.nan
    except Exception:
        pchem_raw = np.nan
    try:
        pctrl = model.predict(pd.DataFrame([feat_ctrl]))[0] if model is not None else np.nan
    except Exception:
        pctrl = np.nan
    preds_bar_chem_raw.append(pchem_raw)
    preds_bar_ctrl.append(pctrl)
    wt_at_days.append(wt_series_full[d-1])

# apply depletion multiplier at those days
if enable_depletion:
    mult_days = 1.0 - (alpha * np.array(wt_at_days) + beta)
    mult_days = np.clip(mult_days, 0.0, 1.0)
else:
    mult_days = np.ones(len(days_show))
preds_bar_chem = np.array(preds_bar_chem_raw) * mult_days
preds_bar_ctrl = np.array(preds_bar_ctrl)

# experimental stats for selected env & GO (prefer exact GO)
exp_same_go = experiment_df[(experiment_df["environment"]==env) & (np.isclose(experiment_df["GO_pct"], round(go_pct,2)))]
if not exp_same_go.empty:
    exp_stats = exp_same_go.groupby("day")["compressive_strength_MPa"].agg(["mean","std"]).reindex(days_show).fillna(np.nan)
else:
    exp_stats = experiment_df[experiment_df["environment"]==env].groupby("day")["compressive_strength_MPa"].agg(["mean","std"]).reindex(days_show).fillna(np.nan)

fig2, ax2 = plt.subplots(figsize=(9,4))
x = np.arange(len(days_show)); w=0.25
exp_vals = exp_stats["mean"].values if not exp_stats.empty else np.array([np.nan]*len(days_show))
exp_errs = safe_errbars(exp_stats["std"].fillna(np.nan)) if not exp_stats.empty else None
bars_exp = ax2.bar(x - w, exp_vals, w, yerr=exp_errs, capsize=4, label="Experiment", color="#f7c873", edgecolor='k')
bars_model_chem = ax2.bar(x, preds_bar_chem, w, label=f"Model depleted ({env})", color="#d95f02", edgecolor='k')
bars_model_ctrl = ax2.bar(x + w, preds_bar_ctrl, w, label="Model control", color="#2b7a78", edgecolor='k')
def add_labels(bars):
    for b in bars:
        h = b.get_height()
        if np.isnan(h): continue
        ax2.text(b.get_x()+b.get_width()/2, h+0.6, f"{h:.2f}", ha='center', va='bottom', fontsize=9)
add_labels(bars_exp); add_labels(bars_model_chem); add_labels(bars_model_ctrl)
ax2.set_xticks(x); ax2.set_xticklabels([f"{d}d" for d in days_show])
ax2.set_ylabel("Compressive Strength (MPa)")
ax2.set_title(f"Experiment vs Model (depleted) vs Control model — GO {go_pct:.3f}%")
ax2.legend(); ax2.grid(axis="y", alpha=0.2, linestyle="--")
st.pyplot(fig2)
buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", bbox_inches="tight"); buf2.seek(0)
st.download_button("Download grouped comparison (PNG)", buf2, file_name=f"grouped_comp_{env}_GO{go_pct:.3f}.png", mime="image/png")

# ---------------------------
# Parity + saved charts + validation table
# ---------------------------
st.markdown("## Validation & Saved Visuals")
colp, colb = st.columns([1,2])
with colp:
    st.subheader("Parity (saved)")
    if os.path.exists(PARITY_IMG):
        st.image(PARITY_IMG, use_column_width=True)
        with open(PARITY_IMG, "rb") as f:
            st.download_button("Download parity image", f.read(), file_name="parity_retrained_pdf.png", mime="image/png")
    else:
        st.write("Parity image not found (run retrain scripts).")

with colb:
    st.subheader("Saved bar charts")
    if os.path.exists(BAR_DIR):
        imgs = [os.path.join(BAR_DIR,f) for f in sorted(os.listdir(BAR_DIR)) if f.lower().endswith(".png")]
        if imgs:
            for im in imgs:
                st.image(im, width=280)
                with open(im, "rb") as f:
                    st.download_button(os.path.basename(im), f.read(), file_name=os.path.basename(im), mime="image/png")
        else:
            st.write("No bar charts saved.")
    else:
        st.write("Bar charts folder not found.")

st.markdown("### Validation table (tuned)")
if validation_tuned_df is not None:
    display_cols = [c for c in ["environment","GO_pct","day","exp_mean_MPa","exp_std_MPa","model_pred_MPa","abs_err","rel_err_pct"] if c in validation_tuned_df.columns]
    st.dataframe(validation_tuned_df[display_cols].sort_values(["environment","GO_pct","day"]))
    st.download_button("Download tuned validation CSV", validation_tuned_df.to_csv(index=False).encode("utf-8"), file_name="validation_table_model_vs_experiment_exposure_tuned.csv", mime="text/csv")
else:
    st.write("Tuned validation CSV not found.")

# Show performance CSV if present
st.markdown("### Performance summary")
if os.path.exists(PERF_TUNED):
    try:
        perft = pd.read_csv(PERF_TUNED)
        st.table(perft.T)
        st.download_button("Download tuned perf CSV", open(PERF_TUNED,"rb").read(), file_name=os.path.basename(PERF_TUNED), mime="text/csv")
    except Exception:
        st.write("Could not read tuned perf CSV.")
elif os.path.exists(PERF_BASE):
    try:
        perfb = pd.read_csv(PERF_BASE)
        st.table(perfb.T)
        st.download_button("Download perf CSV", open(PERF_BASE,"rb").read(), file_name=os.path.basename(PERF_BASE), mime="text/csv")
    except Exception:
        st.write("Could not read perf CSV.")
else:
    st.write("No performance CSV found.")

# ---------------------------
# Literature titles & standards
# ---------------------------
st.markdown("## Literature & Standards")
if show_journals and os.path.exists(JOURNALS_TXT):
    with open(JOURNALS_TXT, "r", encoding="utf-8") as f:
        jt = [ln.strip() for ln in f.readlines() if ln.strip()]
    st.write("Validating journal titles:")
    for j in jt:
        st.write("- " + j)
else:
    st.write("Journal titles file not found or hidden.")

st.write("- IS 4031 — Methods of physical tests for hydraulic cement.")
st.write("- IS 516 — Methods of tests for strength of concrete (cube testing).")

# ---------------------------
# Final downloads & notes
# ---------------------------
st.markdown("---")
st.write("Download model files:")
if os.path.exists(TUNED_MODEL):
    with open(TUNED_MODEL, "rb") as f:
        st.download_button("Download tuned exposure model (.pkl)", f.read(), file_name=os.path.basename(TUNED_MODEL), mime="application/octet-stream")
elif os.path.exists(EXPOSURE_MODEL_FALLBACK):
    with open(EXPOSURE_MODEL_FALLBACK, "rb") as f:
        st.download_button("Download exposure model (.pkl)", f.read(), file_name=os.path.basename(EXPOSURE_MODEL_FALLBACK), mime="application/octet-stream")
elif os.path.exists(RETRAIN_MODEL_OLD):
    with open(RETRAIN_MODEL_OLD, "rb") as f:
        st.download_button("Download retrained base model (.pkl)", f.read(), file_name=os.path.basename(RETRAIN_MODEL_OLD), mime="application/octet-stream")
elif os.path.exists(DEFAULT_MODEL):
    with open(DEFAULT_MODEL, "rb") as f:
        st.download_button("Download default continuous model (.pkl)", f.read(), file_name=os.path.basename(DEFAULT_MODEL), mime="application/octet-stream")
else:
    st.write("No model file available for download.")
# ---------------------------------------------------------
# Thesis / Report Package (ZIP Download)
# ---------------------------------------------------------
import zipfile
from io import BytesIO

st.markdown("## 📦 Download Full Thesis/Report Package")

# Build ZIP on the fly
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
    # Include validation CSV
    if validation_tuned_df is not None:
        z.writestr("validation_table_model_vs_experiment_exposure_tuned.csv",
                   validation_tuned_df.to_csv(index=False))

    # Include performance CSV
    if os.path.exists(PERF_TUNED):
        with open(PERF_TUNED, "rb") as f:
            z.writestr(os.path.basename(PERF_TUNED), f.read())

    # Include parity plot
    if os.path.exists(PARITY_IMG):
        with open(PARITY_IMG, "rb") as f:
            z.writestr("parity_plot.png", f.read())

    # Include bar charts
    if os.path.exists(BAR_DIR):
        for fname in os.listdir(BAR_DIR):
            if fname.lower().endswith(".png"):
                with open(os.path.join(BAR_DIR, fname), "rb") as f:
                    z.writestr(f"bar_charts/{fname}", f.read())

    # Include experimental dataset
    if os.path.exists(EXPCSV):
        with open(EXPCSV, "rb") as f:
            z.writestr("experimental_dataset_used_from_pdf.csv", f.read())

    # Include model file
    if os.path.exists(TUNED_MODEL):
        with open(TUNED_MODEL, "rb") as f:
            z.writestr("gb_model_exposure_tuned.pkl", f.read())

    # Summary notes
    z.writestr("README.txt",
               "Thesis/Report Package\n"
               "- Contains validation table, performance metrics, model, and charts\n"
               "- Suitable for thesis appendices and viva presentations\n"
               "- Generated by Streamlit app\n")

zip_buffer.seek(0)

st.download_button(
    label="📥 Download Thesis/Report ZIP (C3)",
    data=zip_buffer.getvalue(),
    file_name="thesis_report_package_C3.zip",
    mime="application/zip"
)

st.caption("Model predictions are generated with the tuned exposure-trained model when available. Depletion mapping is data-driven (alpha mapping) and applied after model baseline prediction; this shows realistic degradation and GO resistance for visualization and validation.")
