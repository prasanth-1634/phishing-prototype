# app.py
import streamlit as st
import joblib
import pandas as pd
from train import extract_features

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(
    page_title="Phishing Detection Prototype",
    page_icon="🕵‍♂",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center; color:#3b82f6;'>🕵‍♂ Phishing URL Detection Prototype</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Enter a website link below to check if it's legitimate or a phishing attempt.</p>",
    unsafe_allow_html=True
)

# ------------------------------
# Load model
# ------------------------------
try:
    model = joblib.load("model.joblib")
except:
    st.error("Model not found. Please run train.py first to train and save the model.")
    st.stop()

# ------------------------------
# Input section
# ------------------------------
st.write("---")
url = st.text_input("🔗 Enter the website URL to check:", value="https://www.google.com")
check_button = st.button("🔍 Check Now")

# ------------------------------
# Prediction
# ------------------------------
if check_button:
    with st.spinner("Analyzing URL..."):
        features = extract_features(url)
        X = pd.DataFrame([features])
        prob = model.predict_proba(X)[0][1]
        label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"

        # Show result with color and emoji
        if label == "PHISHING":
            st.error(f"🚨 This website seems *PHISHING!\n\nConfidence:* {prob:.2f}")
        else:
            st.success(f"✅ This website seems *LEGITIMATE.\n\nConfidence:* {1-prob:.2f}")

        # Show features used for prediction (optional)
        with st.expander("🔬 View Technical Details"):
            st.json(features)

# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; font-size:14px; color:gray;'>Developed by <b>J.Prasanth</b> | Ideathon 2025 | Using Python + Streamlit</p>",
    unsafe_allow_html=True

)

