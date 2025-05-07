import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import gdown

# URL file di Google Drive (ganti dengan URL Anda)
url = 'https://drive.google.com/uc?id=11QP97lrige73cOv9wn6g2ren5DJdzFg-'
output = 'CVD_cleaned.csv'

# Mengunduh file CSV dari Google Drive
gdown.download(url, output, quiet=False)

# Load the dataset
df = pd.read_csv('CVD_cleaned.csv')

# Mengonversi 'No' dan 'Yes' di kolom target menjadi angka 0 dan 1
df['Heart_Disease'] = df['Heart_Disease'].map({'No': 0, 'Yes': 1})

# Pilih fitur yang akan digunakan untuk pelatihan
features = ['BMI', 'Green_Vegetables_Consumption', 'Weight_(kg)', 
            'Fruit_Consumption', 'FriedPotato_Consumption', 
            'Height_(cm)', 'Alcohol_Consumption']

# Memisahkan fitur dan target
X = df[features]  # Input untuk model
y = df['Heart_Disease']  # Target variabel (yang akan diprediksi)

# Menghitung class weights untuk menangani ketidakseimbangan kelas
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Melatih model RandomForest
rf_model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
rf_model.fit(X, y)

# Streamlit App
st.title("Heart Disease Prediction App")

# User input form
with st.form("user_input_form"):
    st.header("Enter Your Details")

    # Collect user input for relevant features
    height = st.number_input("Enter your height (in cm):", min_value=100, max_value=250, step=1)
    weight = st.number_input("Enter your weight (in kg):", min_value=30.0, max_value=200.0, step=0.1, format="%.2f")
    
    # Calculate BMI based on height and weight
    bmi = weight / ((height / 100) ** 2)
    st.write(f"Your calculated BMI is: {bmi:.2f}")
    
    # Add input fields for other relevant columns
    green_vegetables_consumption = st.number_input("Enter the amount of green vegetables you consume weekly (per 100gram):", min_value=0, max_value=100, step=1)
    fruit_consumption = st.number_input("Enter the amount of fruit you consume weekly (per 100gram):", min_value=0, max_value=100, step=1)
    fried_potato_consumption = st.number_input("Enter the amount of fried potatoes you consume weekly (per 100gram):", min_value=0, max_value=100, step=1)
    alcohol_consumption = st.number_input("Enter the number of alcohol units consumed per week:", min_value=0, max_value=50, step=1)
    
    # Submit button
    submit_button = st.form_submit_button("Predict Heart Disease")

# Prediction logic
if submit_button:
    # Collect user data (only relevant features)
    user_data = {
        'BMI': [bmi],
        'Green_Vegetables_Consumption': [green_vegetables_consumption],
        'Weight_(kg)': [weight],
        'Fruit_Consumption': [fruit_consumption],
        'FriedPotato_Consumption': [fried_potato_consumption],
        'Height_(cm)': [height],
        'Alcohol_Consumption': [alcohol_consumption]
    }

    # Create DataFrame for user data
    user_df = pd.DataFrame(user_data)

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
