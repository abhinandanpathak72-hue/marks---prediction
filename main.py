import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- Load the saved model and label encoders ---
@st.cache_resource
def load_resources():
    try:
        with open('best_xgb_model_rand.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Error: 'best_xgb_model_rand.pkl' not found. Please ensure it's in the same directory.")
        st.stop()

    try:
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    except FileNotFoundError:
        st.error("Error: 'label_encoders.pkl' not found. Please ensure it's in the same directory.")
        st.stop()

    return model, label_encoders

model, label_encoders = load_resources()

# --- Streamlit Application Layout ---
st.set_page_config(page_title="Exam Score Predictor")
st.title("Exam Score Prediction App")
st.write("Enter the student's details to predict their exam score.")

# --- Input Fields ---
st.header("Student Information")

# Numerical features
study_hours = st.slider("Study Hours", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
sleep_hours = st.slider("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.1)

# Categorical features (that were part of X)
internet_access_options = ['yes', 'no']
internet_access = st.selectbox("Internet Access", internet_access_options)

sleep_quality_options = ['poor', 'average', 'good']
sleep_quality = st.selectbox("Sleep Quality", sleep_quality_options)

study_method_options = ['coaching', 'online videos', 'self-study', 'group study']
study_method = st.selectbox("Study Method", study_method_options)

facility_rating_options = ['low', 'medium', 'high']
facility_rating = st.selectbox("Facility Rating", facility_rating_options)


if st.button("Predict Exam Score"):
    # Create a dictionary for the input features
    input_data = {
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'internet_access': internet_access,
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'study_method': study_method,
        'facility_rating': facility_rating
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply label encoding for categorical features
    # Only apply to the features that the model expects and were encoded
    for col in ['internet_access', 'sleep_quality', 'study_method', 'facility_rating']:
        if col in label_encoders:
            # Use the stored LabelEncoder to transform the input
            # Ensure the input is within the known classes of the encoder
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"Error encoding '{col}': {e}. Please check the selected value.")
                st.stop()

    # Make prediction
    try:
        prediction = model.predict(input_df)
        st.success(f"The predicted exam score is: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")

