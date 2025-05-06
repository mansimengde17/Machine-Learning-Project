import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("random_forest_tuned.pickle")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Risk Predictor")
st.markdown("Enter the following health indicators to assess your diabetes risk.")

# Input Fields
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
Smoker = st.number_input("Smoker (0 = No, 1 = Yes)", 0, 1)
Stroke = st.number_input("Stroke (0 = No, 1 = Yes)", 0, 1)
HeartDiseaseorAttack = st.number_input("Heart Disease or Attack (0 = No, 1 = Yes)", 0, 1)
PhysActivity = st.number_input("Physical Activity (0 = No, 1 = Yes)", 0, 1)
Fruits = st.number_input("Fruits Consumption (0 = No, 1 = Yes)", 0, 1)
Veggies = st.number_input("Vegetables Consumption (0 = No, 1 = Yes)", 0, 1)
HvyAlcoholConsump = st.number_input("Heavy Alcohol Consumption (0 = No, 1 = Yes)", 0, 1)
AnyHealthcare = st.number_input("Access to Healthcare (0 = No, 1 = Yes)", 0, 1)
NoDocbcCost = st.number_input("Couldn’t See Doctor Due to Cost (0 = No, 1 = Yes)", 0, 1)
GenHlth = st.number_input("General Health (1 = Excellent to 5 = Poor)", 1, 5)
MentHlth = st.number_input("Mental Health Days (0–30)", 0, 30)
PhysHlth = st.number_input("Physical Health Days (0–30)", 0, 30)
DiffWalk = st.number_input("Difficulty Walking (0 = No, 1 = Yes)", 0, 1)
Sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1)
Age = st.number_input("Age Category (1–13)", 1, 13)

# Prediction
if st.button("Predict Diabetes Risk"):
    user_input = np.array([[BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity,
                            Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
                            NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk,
                            Sex, Age]])

    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]

    label = "At Risk of Diabetes" if prediction == 1 else "No Diabetes Detected"
    st.success(f"Prediction: {label}")