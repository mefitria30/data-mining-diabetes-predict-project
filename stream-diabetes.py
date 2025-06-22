import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# load dan latih ulang scaler
diabetes_dataset = pd.read_csv('diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
scaler = StandardScaler()
scaler.fit(X)

# page config
st.set_page_config(page_title="Prediksi Diabetes", page_icon="💉", layout="centered")

# judul dengan markdown styled
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 Prediksi Diabetes dengan Data Mining</h1>", unsafe_allow_html=True)
st.write("Masukkan data pasien untuk memprediksi kemungkinan diabetes berdasarkan model yang telah dilatih.")

# Form Input
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input('🍼 Pregnancies', min_value=0)
        BloodPressure = st.number_input('💓 BloodPressure', min_value=0)
        Insulin = st.number_input('💉 Insulin', min_value=0)
        DiabetesPedigreeFunction = st.number_input('🧬 DiabetesPedigreeFunction', min_value=0.0, format="%.6f")

    with col2:
        Glucose = st.number_input('🍬 Glucose', min_value=0)
        SkinThickness = st.number_input('📏 SkinThickness', min_value=0)
        BMI = st.number_input('⚖️ BMI', min_value=0.0, format="%.2f")
        Age = st.number_input('🎂 Age', min_value=0)

    submit = st.form_submit_button('🔎 Prediksi Sekarang')

# Prediksi
if submit:
    user_input = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                BMI, DiabetesPedigreeFunction, Age]], columns=X.columns)
    user_input_scaled = scaler.transform(user_input)
    diab_prediction = diabetes_model.predict(user_input_scaled)

    if diab_prediction[0] == 1:
        st.markdown("<h3 style='color: red;'>⚠️ Hasil: Pasien kemungkinan <u>TERKENA</u> diabetes.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>✅ Hasil: Pasien kemungkinan <u>TIDAK TERKENA</u> diabetes.</h3>", unsafe_allow_html=True)
