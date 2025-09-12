import os
import io
import json
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(page_title="Anti-Aging MVP", layout="centered")
st.title("Anti-Aging Epigenetics MVP")

with st.sidebar:
    st.header("Auth")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign up"):
        r = requests.post(f"{API_URL}/signup", json={"username": username, "password": password})
        st.toast("Signed up" if r.ok else f"Signup failed: {r.text}")
    if st.button("Login"):
        r = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
        if r.ok:
            token = r.json().get("access_token")
            st.session_state["token"] = token
            st.success("Logged in")
        else:
            st.error(f"Login failed: {r.text}")

headers = {}
if "token" in st.session_state:
    headers["Authorization"] = f"Bearer {st.session_state['token']}"

st.header("Upload Genetic CSV")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded and st.button("Upload"):
    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
    r = requests.post(f"{API_URL}/upload-genetic", headers=headers, files=files)
    if r.ok:
        st.success("Uploaded")
    else:
        st.error(r.text)

st.header("Submit Habits")
habits = st.text_area("Habits JSON", value=json.dumps({"sleep_hours": 7, "exercise_minutes": 30, "smoking": 0}, indent=2))
if st.button("Submit Habits"):
    try:
        data = json.loads(habits)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
    else:
        r = requests.post(f"{API_URL}/submit-habits", headers=headers, json=data)
        st.write(r.json() if r.ok else r.text)

st.header("Predict")
model_type = st.selectbox("Model", options=["rf", "nn"], index=0)
if st.button("Run Prediction"):
    r = requests.post(f"{API_URL}/predict", headers=headers, params={"model_type": model_type})
    if r.ok:
        out = r.json()
        st.metric("Predicted Biological Age", f"{out.get('predicted_age')}")
        st.subheader("Explanation")
        st.json(out.get("explanation"))
    else:
        st.error(r.text)
