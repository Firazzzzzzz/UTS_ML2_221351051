import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import os

# ==== Custom CSS ====
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-color: #f3f4f6;
            padding: 2rem;
            max-width: 880px;
            margin: auto;
        }
        .header {
            background: linear-gradient(to right, #4facfe, #00f2fe);
            padding: 1.8rem 1rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            font-size: 2rem;
            font-weight: 600;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
        }
        .subtext {
            text-align: center;
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .form-box {
            background: #ffffff;
            padding: 2rem;
            border-radius: 14px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
        }
        .result-box {
            margin-top: 2rem;
            background: #ffffff;
            padding: 1.8rem;
            border-radius: 14px;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
            font-size: 1.2rem;
        }
        .high-risk {
            color: #e74c3c;
            font-weight: 600;
        }
        .low-risk {
            color: #2ecc71;
            font-weight: 600;
        }
        .stButton button {
            background-color: #0077cc;
            color: white;
            padding: 0.65rem 1.3rem;
            border-radius: 8px;
            font-size: 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #005fa3;
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# ==== Load Model ====
@st.cache_resource
def load_model_and_scaler():
    st.write("🔄 Memuat model dan scaler...")
    
    # Cek file scaler
    if not os.path.exists("loan_scaler.pkl"):
        return None, None, "❌ File scaler 'loan_scaler.pkl'_
