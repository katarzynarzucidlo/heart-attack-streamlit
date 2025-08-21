# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="🫀", layout="centered")

# -----------------------------
# Load trained model/pipeline
# -----------------------------
MODEL_PATH = "heart_attack_rf_pipeline.joblib"

@st.cache_resource
def load_model(path: str):
    model = joblib.load(path)
    return model

model = load_model(MODEL_PATH)

# Try to discover expected feature names from the fitted estimator (if available)
EXPECTED = getattr(model, "feature_names_in_", None)

st.title("🫀 Heart Attack Risk Predictor")
st.caption("Random Forest (optimized). Provide patient info to estimate risk.")

st.divider()

# ----------------------------------------
# INPUT WIDGETS (raw clinical features)
# If your pipeline does preprocessing, keep these as the "raw" names.
# ----------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
    trtbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=130, step=1)
    thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)

with col2:
    sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], index=1, format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type (0–3)", options=[0, 1, 2, 3], index=2)
    exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", options=[0, 1], index=0)
    slope = st.selectbox("Slope of ST Segment (0–2)", options=[0, 1, 2], index=2)
    ca = st.selectbox("Number of Major Vessels (0–3)", options=[0, 1, 2, 3], index=0)
    thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed defect, 3=Reversible defect)", options=[1, 2, 3], index=2)

# -----------------------------------------------------------------------
# If your MODEL expects engineered column names, RENAME THEM HERE.
# Example: You mentioned you dropped 'chol' and engineered:
#   'trtbps'  ->  'trtbps_winsorize'
#   'oldpeak' ->  'oldpeak_winsorize_sqrt'
#
# Edit the mapping below to match the exact column names used in training.
# If your pipeline handles these transforms internally, leave this EMPTY.
# -----------------------------------------------------------------------
RENAME_MAP = {
    # "trtbps": "trtbps_winsorize",
    # "oldpeak": "oldpeak_winsorize_sqrt",
}

# ----------------------------------------
# Build a single-row DataFrame
# ----------------------------------------
raw_row = {
    "age": age,
    "trtbps": trtbps,      # will be renamed if needed
    "thalach": thalach,
    "oldpeak": oldpeak,    # will be renamed if needed
    "sex": sex,
    "cp": cp,
    "exang": exang,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}

df_input = pd.DataFrame([raw_row])

# Apply optional renaming if you trained on engineered column names directly
if RENAME_MAP:
    df_input = df_input.rename(columns=RENAME_MAP)

# If model exposes feature_names_in_, align/order columns;
# otherwise, rely on pipeline/estimator to handle it.
if EXPECTED is not None:
    # Only keep columns the model expects; add any missing as zeros
    missing = [c for c in EXPECTED if c not in df_input.columns]
    for m in missing:
        df_input[m] = 0
    df_input = df_input[EXPECTED]

st.write("**Model input preview**")
st.dataframe(df_input, use_container_width=True)

st.divider()

threshold = st.slider("Decision threshold (for class label)", 0.10, 0.90, 0.50, 0.05)

if st.button("Predict risk", type="primary"):
    try:
        # Most sklearn classifiers implement predict_proba; if not, fall back to decision_function
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[:, 1][0]
        else:
            # Scale decision_function to 0-1 if needed (simple logistic mapping fallback)
            df = model.decision_function(df_input)[0]
            proba = 1 / (1 + np.exp(-df))

        pred = int(proba >= threshold)

        st.subheader("Result")
        st.metric("Estimated risk (probability)", f"{proba:.2%}")
        st.write(f"**Predicted class @ threshold {threshold:.2f}:** {'High Risk (1)' if pred==1 else 'Low Risk (0)'}")

        st.info(
            "This tool is for educational use. Clinical decisions require a full clinical workflow and physician oversight."
        )

    except Exception as e:
        st.error(
            "Prediction failed. This usually means the model expects different column names or preprocessing. "
            "Edit the RENAME_MAP (above) to match your training columns, or ensure your saved object is a full Pipeline."
        )
        st.exception(e)

st.caption("Model file: `heart_attack_rf_pipeline.joblib`  •  App by Streamlit")
