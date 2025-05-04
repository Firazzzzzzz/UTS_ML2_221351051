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
    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("❌ File scaler 'scaler.pkl' tidak ditemukan.")
        st.stop()

    if not os.path.exists("loan_default.tflite"):
        st.error("❌ Model 'loan_default.tflite' tidak ditemukan.")
        st.stop()

    interpreter = tf.lite.Interpreter(model_path="loan_default.tflite")
    interpreter.allocate_tensors()
    return scaler, interpreter

scaler, interpreter = load_model_and_scaler()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== Header ====
st.markdown('<div class="header">📊 Prediksi Risiko Gagal Bayar Pinjaman</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Masukkan informasi peminjam untuk melihat kemungkinan gagal bayar berdasarkan model machine learning.</div>', unsafe_allow_html=True)

# ==== Input Form ====
with st.form("loan_form"):
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia Pemohon (tahun)", 18, 100, 30)
        income = st.number_input("Pendapatan Bulanan (juta)", 0.0, 1000.0, 10.0)
    with col2:
        loan_amount = st.number_input("Jumlah Pinjaman (juta)", 0.0, 1000.0, 100.0)
        loan_term = st.number_input("Durasi Pinjaman (bulan)", 6, 360, 60)

    submitted = st.form_submit_button("Lakukan Prediksi")
    st.markdown('</div>', unsafe_allow_html=True)

# ==== Output Prediction ====
if submitted:
    try:
        input_data = np.array([[age, income, loan_amount, loan_term]])
        input_scaled = scaler.transform(input_data).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        probability = float(prediction[0][0])

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        if probability > 0.5:
            st.markdown(
                f"<p class='high-risk'>⚠️ Hasil Prediksi: <b>Risiko Tinggi</b> gagal bayar — <b>{probability:.2%}</b></p>",
                unsafe_allow_html=True
            )
            st.progress(min(probability, 1.0))
        else:
            st.markdown(
                f"<p class='low-risk'>✅ Hasil Prediksi: <b>Risiko Rendah</b> gagal bayar — <b>{probability:.2%}</b></p>",
                unsafe_allow_html=True
            )
            st.progress(min(probability, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat melakukan prediksi: {e}")
