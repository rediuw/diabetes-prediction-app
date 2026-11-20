import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="Diabetes Risk Classifier", page_icon="🩺")


@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")


model = load_model()

st.title("Diabetes Risk Classifier")
st.caption("Educational machine learning project. This is NOT a medical diagnosis tool.")

left, right = st.columns(2)
with left:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, 120.0, help="Enter 0 if unknown")
    blood_pressure = st.number_input("Blood pressure (mm Hg)", 0.0, 200.0, 70.0, help="Enter 0 if unknown")
    skin = st.number_input("Skin thickness (mm)", 0.0, 100.0, 20.0, help="Enter 0 if unknown")
with right:
    insulin = st.number_input("Insulin (mu U/ml)", 0.0, 900.0, 80.0, help="Enter 0 if unknown")
    bmi = st.number_input("BMI", 0.0, 70.0, 28.0, help="Enter 0 if unknown")
    dpf = st.number_input("Diabetes pedigree function", 0.0, 2.5, 0.4, step=0.01)
    age = st.number_input("Age (years)", 18, 100, 35)

if st.button("Predict"):
    row = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age]],
                       columns=FEATURES)
    row[ZERO_MEANS_MISSING] = row[ZERO_MEANS_MISSING].replace(0, np.nan)

    prob = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated probability of diabetes", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= 0.5:
        st.error("Higher risk: the model classifies this profile as likely diabetic.")
    else:
        st.success("Lower risk: the model classifies this profile as likely non-diabetic.")
    st.caption("Please consult a healthcare professional for any real medical concern.")
