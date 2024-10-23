import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained deep learning model
model = tf.keras.models.load_model(r'C:\Users\venka\Documents\SEM 3\DS PRO\DL_LLM\DL_model.h5')

# Load scaler (fit it with training data when creating the model)
scaler = StandardScaler()

# Function to preprocess user input
def preprocess_input(data):
    data_scaled = scaler.transform(data)
    return np.array(data_scaled)

# Streamlit UI setup
st.title("Diabetes Prediction Application")

# Input fields for user-provided features
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=1, max_value=100, step=1)
bmi = st.number_input("Enter BMI", min_value=0.0, step=0.1)
blood_pressure = st.number_input("Enter Blood Pressure", min_value=0.0, step=0.1)
glucose = st.number_input("Enter Glucose Level", min_value=0.0, step=0.1)

# Predict button
to_predict = st.button("Make Prediction")
if to_predict:
    # Encode the gender feature
    gender_value = 1 if gender == "Male" else 0

    # Prepare input data
    input_features = [[gender_value, age, bmi, blood_pressure, glucose]]
    processed_data = preprocess_input(input_features)

    # Model prediction
    prediction_result = model.predict(processed_data)[0]

    # Display prediction
    if prediction_result > 0.5:
        st.success("The model predicts that the patient is likely to have diabetes.")
    else:
        st.success("The model predicts that the patient is unlikely to have diabetes.")
