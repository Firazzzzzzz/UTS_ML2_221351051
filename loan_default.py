import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Loan Default", layout="centered", page_icon="💸")

# Load model dan scaler
scaler = joblib.load('scaler.pkl')
interpreter = tf.lite.Interpreter(model_path="loan_default.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.markdown("<h1 style='text-align: center;'> Prediksi Loan Default</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data peminjam untuk mengetahui kemungkinan gagal bayar pinjaman.</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout input dalam dua kolom
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Umur", min_value=18, max_value=100, value=30)
    loan_amount = st.number_input("💰 Jumlah Pinjaman (Rp)", min_value=0, value=10000000)

with col2:
    income = st.number_input("📈 Penghasilan Bulanan (Rp)", min_value=0, value=5000000)
    loan_term = st.number_input("🕒 Lama Pinjaman (bulan)", min_value=1, max_value=360, value=12)

credit_history = st.selectbox("📊 Riwayat Kredit", options=["Buruk", "Sedang", "Baik"])

# Encoding fitur kategorikal
credit_mapping = {"Buruk": 0, "Sedang": 1, "Baik": 2}
credit_encoded = credit_mapping[credit_history]

# Tombol prediksi
if st.button("🔍 Prediksi Gagal Bayar"):
    # Perhatikan perubahan pada baris ini. Kita hanya memilih 4 fitur untuk scaling.
    input_data = np.array([[age, income, loan_amount, loan_term]])

    try:
        input_scaled = scaler.transform(input_data).astype(np.float32)
    except ValueError as e:
        st.error(f"❌ Error: {e}")
        st.stop()

    # Pastikan input ke model TFLite memiliki bentuk yang benar
    input_reshaped = input_scaled.reshape(1, input_scaled.shape[1])
    interpreter.set_tensor(input_details[0]['index'], input_reshaped)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Interpretasi hasil
    predicted_prob = prediction[0][0]
    predicted_class = int(np.round(predicted_prob))

    st.markdown("---")
    st.subheader("📈 Hasil Prediksi")

    if predicted_class == 1:
        st.error(f"⚠️ Risiko tinggi untuk gagal bayar! (Probabilitas: {predicted_prob:.2%})")
    else:
        st.success(f"✅ Kemungkinan besar tidak gagal bayar. (Probabilitas: {predicted_prob:.2%})")