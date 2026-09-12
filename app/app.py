import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict

st.set_page_config(page_title="Rainfall Prediction System", page_icon="🌧️", layout="centered")

MODEL_DIR = "models"


# ------------------------------------------------------------------
# Load artifacts once (cached across reruns)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    selected_idx = joblib.load(os.path.join(MODEL_DIR, "selected_feature_indices.joblib"))
    schema = joblib.load(os.path.join(MODEL_DIR, "input_schema.joblib"))
    base_model = joblib.load(os.path.join(MODEL_DIR, "svm_baseline.joblib"))
    pso_model = joblib.load(os.path.join(MODEL_DIR, "svm_pso_selected.joblib"))
    return preprocessor, feature_names, selected_idx, schema, base_model, pso_model


try:
    preprocessor, feature_names, selected_idx, schema, base_model, pso_model = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model artifacts not found. Make sure the `models/` folder "
        "(produced by the Colab save cell) sits next to this app.py.\n\n"
        f"Details: {e}"
    )
    st.stop()

ALL_RAW = schema["numeric_cols"] + schema["categorical_cols"]


# ------------------------------------------------------------------
# Work out which RAW inputs the PSO model actually needs.
# The 115 encoded columns map back to 22 raw inputs (Location alone
# expands to ~49 one-hot columns). If NONE of a raw input's encoded
# columns were selected by PSO, that input has zero effect on the
# PSO model and does not need to be collected.
# ------------------------------------------------------------------
@st.cache_data
def raw_inputs_needed_by_pso(_feature_names, _selected_idx, _cat_cols):
    selected = set(int(i) for i in _selected_idx)

    def raw_source(name):
        body = name.split("__", 1)[1] if "__" in name else name
        for c in _cat_cols:
            if body.startswith(c + "_"):
                return c
        return body

    groups = defaultdict(lambda: {"total": 0, "kept": 0})
    for i, n in enumerate(_feature_names):
        src = raw_source(n)
        groups[src]["total"] += 1
        if i in selected:
            groups[src]["kept"] += 1

    needed = sorted([s for s, g in groups.items() if g["kept"] > 0])
    dropped = sorted([s for s, g in groups.items() if g["kept"] == 0])
    return needed, dropped, dict(groups)


PSO_NEEDED, PSO_DROPPED, GROUP_INFO = raw_inputs_needed_by_pso(
    feature_names, selected_idx, tuple(schema["categorical_cols"])
)

# Friendly labels / grouping for the form
LABELS = {
    "MinTemp": "Minimum temperature (°C)", "MaxTemp": "Maximum temperature (°C)",
    "Temp9am": "Temperature at 9am (°C)", "Temp3pm": "Temperature at 3pm (°C)",
    "Rainfall": "Rainfall today (mm)", "Evaporation": "Evaporation (mm)",
    "Sunshine": "Sunshine (hours)", "WindGustSpeed": "Wind gust speed (km/h)",
    "WindSpeed9am": "Wind speed at 9am (km/h)", "WindSpeed3pm": "Wind speed at 3pm (km/h)",
    "Humidity9am": "Humidity at 9am (%)", "Humidity3pm": "Humidity at 3pm (%)",
    "Pressure9am": "Pressure at 9am (hPa)", "Pressure3pm": "Pressure at 3pm (hPa)",
    "Cloud9am": "Cloud cover at 9am (oktas)", "Cloud3pm": "Cloud cover at 3pm (oktas)",
    "Month": "Month", "RainToday": "Did it rain today?",
    "Location": "Location", "WindGustDir": "Wind gust direction",
    "WindDir9am": "Wind direction at 9am", "WindDir3pm": "Wind direction at 3pm",
}

SECTIONS = [
    ("Location & time", ["Location", "Month"]),
    ("Temperature & sunshine", ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "Sunshine", "Evaporation"]),
    ("Humidity, pressure & cloud", ["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm"]),
    ("Wind & rainfall", ["WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm",
                         "WindSpeed9am", "WindSpeed3pm", "Rainfall", "RainToday"]),
]


def render_field(feat, key_prefix):
    """Render one input widget for a raw feature."""
    label = LABELS.get(feat, feat)
    key = f"{key_prefix}_{feat}"

    if feat == "RainToday":
        v = st.selectbox(label, ["No", "Yes"], key=key)
        return 1.0 if v == "Yes" else 0.0
    if feat == "Month":
        return float(st.slider(label, 1, 12, 6, key=key))
    if feat in schema["categorical_cols"]:
        opts = schema["categorical_options"].get(feat, ["N"])
        return st.selectbox(label, opts, key=key)
    lo, hi, med = schema["numeric_ranges"][feat]
    return float(st.number_input(label, value=float(round(med, 1)), key=key))


def render_input_form(fields, key_prefix):
    """Render only the given raw fields, grouped into sensible sections."""
    values = {}
    for title, feats in SECTIONS:
        show = [f for f in feats if f in fields]
        if not show:
            continue
        with st.expander(title, expanded=True):
            cols = st.columns(min(3, len(show)))
            for i, feat in enumerate(show):
                with cols[i % len(cols)]:
                    values[feat] = render_field(feat, key_prefix)
    return values


def build_raw_row(values):
    """
    Assemble a one-row DataFrame containing EVERY raw column the
    preprocessor expects. Fields not collected (because the PSO model
    ignores them) are filled with the training median / modal value.
    Those columns are still encoded, but their encoded outputs are not
    in selected_idx, so they have provably zero effect on the PSO model.
    """
    row = {}
    for c in schema["numeric_cols"]:
        row[c] = values.get(c, schema["numeric_ranges"][c][2])
    for c in schema["categorical_cols"]:
        row[c] = values.get(c, schema["categorical_options"][c][0])
    return pd.DataFrame([row])


def predict_and_show(model, raw_row, use_selected_features):
    X = preprocessor.transform(raw_row)
    X = X.toarray() if hasattr(X, "toarray") else X
    if use_selected_features:
        X = X[:, selected_idx]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    st.divider()
    if pred == 1:
        st.error("🌧️ **Prediction: Rain is expected tomorrow.**")
    else:
        st.success("☀️ **Prediction: No rain expected tomorrow.**")

    if proba is not None:
        c1, c2 = st.columns(2)
        c1.metric("Probability of No Rain", f"{proba[0]*100:.1f}%")
        c2.metric("Probability of Rain", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🌧️ Rainfall Prediction System")
st.caption("PSO Feature Selection + Support Vector Machine — Final Year Project")

page = st.sidebar.radio("Choose model", ["Baseline SVM (all features)", "PSO + SVM (selected features)"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Encoded features**\n\n"
    f"Baseline: {len(feature_names)}\n\n"
    f"PSO-selected: {len(selected_idx)}\n\n"
    f"Reduction: {(1 - len(selected_idx)/len(feature_names))*100:.1f}%"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Raw inputs required**\n\n"
    f"Baseline: {len(ALL_RAW)}\n\n"
    f"PSO: {len(PSO_NEEDED)}"
)

if page == "Baseline SVM (all features)":
    st.header("Baseline Support Vector Machine")
    st.markdown(
        f"Trained on **all {len(feature_names)} encoded features** with no feature "
        f"selection. It serves as the benchmark against which the PSO-enhanced "
        f"model is compared, and requires all **{len(ALL_RAW)}** weather observations."
    )
    values = render_input_form(ALL_RAW, "base")
    if st.button("Predict with Baseline SVM", type="primary", use_container_width=True):
        predict_and_show(base_model, build_raw_row(values), use_selected_features=False)

else:
    st.header("PSO-Enhanced Support Vector Machine")
    st.markdown(
        f"Trained on the **{len(selected_idx)} encoded features** selected by Binary "
        f"Particle Swarm Optimization. Because the discarded columns trace back to "
        f"whole inputs, this model needs only **{len(PSO_NEEDED)} of the {len(ALL_RAW)}** "
        f"observations — that is the practical benefit of feature selection."
    )
    with st.expander(f"ℹ️ {len(PSO_DROPPED)} inputs PSO discarded entirely — you don't need to measure these"):
        st.write(", ".join(LABELS.get(f, f) for f in PSO_DROPPED))
        st.caption(
            "None of the encoded columns derived from these inputs were selected, "
            "so they have zero effect on this model's prediction."
        )
    values = render_input_form(PSO_NEEDED, "pso")
    if st.button("Predict with PSO + SVM", type="primary", use_container_width=True):
        predict_and_show(pso_model, build_raw_row(values), use_selected_features=True)

st.markdown("---")
st.caption("Kwara State University, Malete — Department of Computer Science")
