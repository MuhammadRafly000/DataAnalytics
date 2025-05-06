import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import gdown

# URL file di Google Drive (ganti dengan URL Anda)
url = 'https://drive.google.com/uc?id=11QP97lrige73cOv9wn6g2ren5DJdzFg-'
output = 'CVD_cleaned.csv'

# Mengunduh file
gdown.download(url, output, quiet=False)

# Load the dataset
df = pd.read_csv('CVD_cleaned.csv')

# List of categorical columns that need encoding
categorical_columns = ['General_Health', 'Checkup', 'Exercise', 'Heart_Disease', 'Skin_Cancer', 
                       'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category', 
                       'Smoking_History']

# Create label encoders for categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define feature columns and target column
X = df.drop('Heart_Disease', axis=1)
y = df['Heart_Disease']

# Handle class imbalance by computing class weights
classes = np.array([0, 1])  # Convert the classes parameter to a NumPy array
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Train a classifier (without GridSearchCV for faster response)
rf_model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
rf_model.fit(X, y)

# Streamlit App
st.title("Heart Disease Prediction App")

# User input form
with st.form("user_input_form"):
    st.header("Enter Your Details")

    # Collect user input
    general_health = st.selectbox("How would you rate your general health?", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
    checkup = st.selectbox("When was your last health checkup?", ["Within the past 2 years", "Within the past year", "Within the past 5 years", "Never"])
    exercise = st.selectbox("Do you exercise regularly?", ["Yes", "No"])
    sex = st.selectbox("Select your sex:", ["Male", "Female"])
    age_category = st.selectbox("Select your age category:", ["60-64", "65-69", "70-74", "75-79", "80+"])

    height = st.number_input("Enter your height (in cm):", min_value=100, max_value=250, step=1)
    weight = st.number_input("Enter your weight (in kg):", min_value=30.0, max_value=200.0, step=0.1, format="%.2f")
    
    # Default BMI value (initialized to 0.00 before user input)
    bmi = 0.00
    
    # Calculate BMI based on height and weight if both are provided
    if height > 0 and weight > 0:
        bmi = weight / ((height / 100) ** 2)
    
    # Display the calculated BMI (placed just above the Predict button)
    st.write(f"Your calculated BMI is: {bmi:.2f}")

    # Add input fields for missing columns
    smoking_history = st.selectbox("Do you have a history of smoking?", ["Yes", "No"])
    alcohol_consumption = st.number_input("Enter the number of alcohol units consumed per week:", min_value=0, max_value=50, step=1)
    fruit_consumption = st.number_input("Enter the amount of fruit you consume weekly (in servings):", min_value=0, max_value=100, step=1)
    green_vegetables_consumption = st.number_input("Enter the amount of green vegetables you consume weekly (in servings):", min_value=0, max_value=100, step=1)
    fried_potato_consumption = st.number_input("Enter the amount of fried potatoes you consume weekly (in servings):", min_value=0, max_value=100, step=1)
    
    # Add input fields for Depression, Diabetes, Skin Cancer, Other Cancer, and Arthritis
    depression = st.selectbox("Do you have a history of depression?", ["Yes", "No"])
    diabetes = st.selectbox("Do you have a history of diabetes?", ["Yes", "No"])
    skin_cancer = st.selectbox("Do you have a history of skin cancer?", ["Yes", "No"])
    other_cancer = st.selectbox("Do you have a history of other cancers?", ["Yes", "No"])
    arthritis = st.selectbox("Do you have a history of arthritis?", ["Yes", "No"])
    
    # Submit button
    submit_button = st.form_submit_button("Predict Heart Disease")

# Prediction logic
if submit_button:
    user_data = {
        'General_Health': [general_health],
        'Checkup': [checkup],
        'Exercise': [exercise],
        'Sex': [sex],
        'Age_Category': [age_category],
        'Height_(cm)': [height],
        'Weight_(kg)': [weight],
        'Smoking_History': [smoking_history],
        'Alcohol_Consumption': [alcohol_consumption],
        'Fruit_Consumption': [fruit_consumption],
        'Green_Vegetables_Consumption': [green_vegetables_consumption],
        'FriedPotato_Consumption': [fried_potato_consumption],
        'Depression': [depression],
        'Diabetes': [diabetes],
        'Skin_Cancer': [skin_cancer],
        'Other_Cancer': [other_cancer],
        'Arthritis': [arthritis],
        'BMI': [bmi]  # Add BMI to the input data
    }
    
    user_df = pd.DataFrame(user_data)
    
    # Ensure all columns are present and encoded the same way as during training
    for col in categorical_columns:
        if col in user_df.columns:
            user_df[col] = label_encoders[col].transform(user_df[col])

    # Check if all features match the training data
    missing_columns = set(X.columns) - set(user_df.columns)
    if missing_columns:
        st.error(f"Missing columns: {missing_columns}. Please provide all required inputs.")
    else:
        # Reorder columns to match the training set
        user_df = user_df[X.columns]

        # Make the prediction using the model
        prediction = rf_model.predict(user_df)
        prediction_proba = rf_model.predict_proba(user_df)

        # Display result
        if prediction == 1:
            st.write("You are at risk of heart disease.")
            st.write(f"Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.write("You are not at risk of heart disease.")
            st.write(f"Probability: {prediction_proba[0][0]:.2f}")
