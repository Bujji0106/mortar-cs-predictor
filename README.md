
# Mortar Compressive Strength Prediction App  
### (AI-driven continuous GO% prediction + durability modeling)

This package contains a fully deployable version of your Streamlit web application using your **continuous GO% trained model**.

---

## 📦 Included Files

- `app_streamlit_realdata.py` — Main Streamlit application  
- `gb_model_continuous.pkl` — Trained ML model (continuous GO behavior)  
- `requirements.txt` — Python dependencies  
- `go_effect_spline_samples.csv` — Continuous GO-effect mapping  
- `parametric_fits_realdata.csv` — Parametric model coefficients  
- `predicted_CS_continuousGO_<env>.png` — Continuous-GO prediction curves for each environment  
- `run.sh` — Linux/Mac launcher  
- `run.bat` — Windows launcher  
- `Dockerfile` — Containerized deployment  
- `final_outputs_no_pdf.zip` — Bundled deployment package  

---

## 🚀 Running the App Locally

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Run the app**
```
streamlit run app_streamlit_realdata.py
```

App will start at:
```
http://localhost:8501
```

---

## 🐳 Running with Docker

### **1. Build the container**
```
docker build -t mortar-cs-app .
```

### **2. Run the container**
```
docker run -p 8501:8501 mortar-cs-app
```

Access app at:
```
http://localhost:8501
```

---

## 💡 Features

- Predict compressive strength for **any continuous GO% (0–0.10%)**
- Predict for **any day (1–365)**
- Supports **MgSO₄, NaCl, H₂SO₄, and controlled curing**
- Upload your own CSV to **retrain the model**
- Download predictions as CSV
- Uses parametric durability decay + ML hybrid model
- Validated with full journal titles you provided

---

## 🔧 Notes
- Streamlit automatically reloads when you change the app file.
- The model file must remain in the same directory as the Streamlit app.

---

## ❤️ Credits
This app was custom-developed for your research work in AI-driven cement mortar durability prediction.
