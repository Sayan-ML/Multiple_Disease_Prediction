# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:59:41 2025

@author: sayan
"""

import streamlit as st
import pickle
import os

# Set page config
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="👨‍🦰")

diabetes_model = pickle.load(open(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\diabetes.pkl','rb'))
heart_disease_model = pickle.load(open(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\heart.pkl','rb'))
kidney_disease_model = pickle.load(open(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\kidney.pkl','rb'))

# Sidebar
selected = st.sidebar.radio("Select Prediction Type", [
    'Diabetes Prediction',
    'Heart Disease Prediction',
    'Kidney Disease Prediction'
])

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.image("https://camo.githubusercontent.com/484bb6ba99bf5fbccd82f484c5bcd286edd8c296ab091919ba58310ddabfe3a8/68747470733a2f2f7265732e636c6f7564696e6172792e636f6d2f67726f6865616c74682f696d6167652f75706c6f61642f635f66696c6c2c665f6175746f2c666c5f6c6f7373792c685f3635302c715f6175746f2c775f313038352f76313538313639353638312f4443554b2f436f6e74656e742f6361757365732d6f662d64696162657465732e706e67", width=500)
    st.title("Diabetes Prediction Using Machine Learning")

    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.slider("Number of Pregnancies", 0, 17, 1)
    with col2: Glucose = st.slider("Glucose Level", 0, 200, 100)
    with col3: BloodPressure = st.slider("Blood Pressure", 0, 122, 70)
    with col1: SkinThickness = st.slider("Skin Thickness", 0, 100, 20)
    with col2: Insulin = st.slider("Insulin", 0, 850, 80)
    with col3: BMI = st.slider("BMI", 0.0, 67.1, 25.0)
    with col1: DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5)
    with col2: Age = st.slider("Age", 21, 81, 33)

    diabetes_result = ""
    if st.button("Get Diabetes Prediction"):
        # BMI Encoding
        NewBMI_Underweight = NewBMI_Overweight = NewBMI_Obesity_1 = NewBMI_Obesity_2 = NewBMI_Obesity_3 = 0
        if float(BMI) <= 18.5:
            NewBMI_Underweight = 1
        elif 24.9 < float(BMI) <= 29.9:
            NewBMI_Overweight = 1
        elif 29.9 < float(BMI) <= 34.9:
            NewBMI_Obesity_1 = 1
        elif 34.9 < float(BMI) <= 39.9:
            NewBMI_Obesity_2 = 1
        elif float(BMI) > 39.9:
            NewBMI_Obesity_3 = 1

        # Insulin & Glucose Encoding
        NewInsulinScore_Normal = 1 if 16 <= float(Insulin) <= 166 else 0
        NewGlucose_Low = NewGlucose_Normal = NewGlucose_Overweight = NewGlucose_Secret = 0
        if float(Glucose) <= 70:
            NewGlucose_Low = 1
        elif 70 < float(Glucose) <= 99:
            NewGlucose_Normal = 1
        elif 99 < float(Glucose) <= 126:
            NewGlucose_Overweight = 1
        elif float(Glucose) > 126:
            NewGlucose_Secret = 1

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age, NewBMI_Underweight,
                      NewBMI_Overweight, NewBMI_Obesity_1, NewBMI_Obesity_2, NewBMI_Obesity_3,
                      NewInsulinScore_Normal, NewGlucose_Low, NewGlucose_Normal,
                      NewGlucose_Overweight, NewGlucose_Secret]
        user_input = [float(i) for i in user_input]
        prediction = diabetes_model.predict([user_input])
        diabetes_result = "The person has diabetes" if prediction[0] == 1 else "The person does not have diabetes"
    st.success(diabetes_result)


# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.image("https://www.cell.com/cms/10.1016/j.heliyon.2024.e38731/asset/6f2ba657-763c-4767-8c70-7ae2a9fefdf2/main.assets/gr1_lrg.jpg", width=600)
    st.title("Heart Disease Prediction Using Machine Learning")

    col1, col2, col3 = st.columns(3)
    with col1: age = st.slider("Age", 25, 80, 50)
    with col2: sex = 1 if st.checkbox("Male") else 0
    with col3: cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    with col1: trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    with col2: chol = st.slider("Serum Cholesterol", 100, 600, 240)
    with col3: fbs = 1 if st.checkbox("Fasting Blood Sugar > 120 mg/dl") else 0
    with col1: restecg = st.slider("Resting ECG (0-2)", 0, 2, 1)
    with col2: thalach = st.slider("Max Heart Rate", 65, 205, 150)
    with col3: exang = 1 if st.checkbox("Exercise Induced Angina") else 0
    with col1: oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1)
    with col2: slope = st.slider("Slope (0-2)", 0, 2, 1)
    with col3: ca = st.slider("Major Vessels Colored (0-4)", 0, 4, 0)
    with col1: thal = st.slider("Thalassemia (0-3)", 0, 3, 1)

    heart_disease_result = ""
    if st.button("Get Heart Disease Prediction"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                      exang, oldpeak, slope, ca, thal]
        prediction = heart_disease_model.predict([user_input])
        heart_disease_result = "The person has heart disease" if prediction[0] == 1 else "No heart disease detected"
    st.success(heart_disease_result)

# Kidney Disease Prediction
if selected == 'Kidney Disease Prediction':
    st.image("https://i.ytimg.com/vi/uxMvHSUOZzc/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLCTgane6pyqALB6y0xm1mETMTx_mA", width=500)
    st.title("Kidney Disease Prediction Using Machine Learning")

    col1, col2, col3 = st.columns(3)
    with col1: age = st.slider("Age", 2, 90, 50)
    with col2: blood_pressure = st.slider("Blood Pressure", 50, 180, 80)
    with col3: specific_gravity = st.slider("Specific Gravity", 1.000, 1.030, 1.015, step=0.001)
    with col1: albumin = st.slider("Albumin", 0, 5, 0)
    with col2: sugar = st.slider("Sugar", 0, 5, 0)
    with col3: red_blood_cells = st.selectbox("Red Blood Cells", [0, 1])
    with col1: pus_cell = st.selectbox("Pus Cell", [0, 1])
    with col2: pus_cell_clumps = st.selectbox("Pus Cell Clumps", [0, 1])
    with col3: bacteria = st.selectbox("Bacteria", [0, 1])
    with col1: blood_glucose_random = st.slider("Blood Glucose Random", 50, 500, 120)
    with col2: blood_urea = st.slider("Blood Urea", 10, 200, 40)
    with col3: serum_creatinine = st.slider("Serum Creatinine", 0.0, 15.0, 1.2, step=0.1)
    with col1: sodium = st.slider("Sodium", 100, 160, 135)
    with col2: potassium = st.slider("Potassium", 2.0, 10.0, 4.5, step=0.1)
    with col3: haemoglobin = st.slider("Haemoglobin", 3.0, 17.5, 13.5, step=0.1)
    with col1: packed_cell_volume = st.slider("Packed Cell Volume", 20, 55, 40)
    with col2: white_blood_cell_count = st.slider("WBC Count (1000s)", 2000, 20000, 8000)
    with col3: red_blood_cell_count = st.slider("RBC Count (millions)", 2.0, 7.0, 5.0, step=0.1)
    with col1: hypertension = 1 if st.checkbox("Hypertension") else 0
    with col2: diabetes_mellitus = 1 if st.checkbox("Diabetes Mellitus") else 0
    with col3: coronary_artery_disease = 1 if st.checkbox("CAD") else 0
    with col1: appetite = st.selectbox("Appetite", [0, 1])
    with col2: peda_edema = st.selectbox("Pedal Edema", [0, 1])
    with col3: aanemia = st.selectbox("Anemia", [0, 1])

    kidney_disease_result = ""
    if st.button("Get Kidney Disease Prediction"):
        user_input = [age, blood_pressure, specific_gravity, albumin, sugar,
                      red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                      blood_glucose_random, blood_urea, serum_creatinine, sodium,
                      potassium, haemoglobin, packed_cell_volume,
                      white_blood_cell_count, red_blood_cell_count, hypertension,
                      diabetes_mellitus, coronary_artery_disease, appetite,
                      peda_edema, aanemia]
        prediction = kidney_disease_model.predict([user_input])
        kidney_disease_result = "The person has kidney disease" if prediction[0] == 1 else "No kidney disease detected"
    st.success(kidney_disease_result)
