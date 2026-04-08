import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="centered"
)

st.markdown("""
    <style>
    div.stButton {
        display: flex;
        justify-content: center;
    }

    div.stButton > button {
        background-color: #ff4b4b;
        color: white;
        padding: 12px 60px;
        border-radius: 50px;
        border: none;
        font-weight: bold;
        font-size: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }

    div.stButton > button:{
        background-color: #e63946;
        transform: scale(1.05);
        color: white;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


st.title("❤️ Heart Stroke Risk dection using KNN")
st.markdown("Developed by **Lakshay** | Smart Health Analytics")
st.divider()

# Input
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("### 👤 Patient Profile")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    st.markdown("### 📊 Clinical Stats")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("<br>", unsafe_allow_html=True)


col_center = st.columns([1, 2, 1])

with col_center[1]:
    predict_btn = st.button("RUN DIAGNOSIS")

# Prediction Logic
if predict_btn:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expected_columns]

    # Scale + Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Output
    st.divider()
    if prediction == 1:
        st.error("### ⚠️ Result: High Risk Detected")
        st.write("The analysis indicates clinical markers associated with heart disease risk.")
    else:
        st.success("### ✅ Result: Low Risk Detected")
        st.write("Your health markers are currently within a lower risk range.")