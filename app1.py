# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:59:41 2025

@author: sayan
"""

import streamlit as st
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="👨‍🦰")

diabetes_model = pickle.load(open(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\diabetes.pkl','rb'))
heart_disease_model = pickle.load(open(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\heart.pkl','rb'))
kidney_disease_model = pickle.load(open(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\kidney.pkl','rb'))
brain_tumor_model = load_model(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\resnet_ft.keras')
eye_disease_model = load_model(r'C:\Users\sayan\OneDrive\Desktop\Multiple disease prediction\resnet_ft.keras')
# Sidebar
selected = st.sidebar.radio("Select Prediction Type", [
    'Diabetes Prediction',
    'Heart Disease Prediction',
    'Kidney Disease Prediction',
    'Brain Tumor Detection using MRI Images',
    'Eye Disease Prediction using OCT Images'
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
    
if selected == 'Brain Tumor Detection using MRI Images':
    st.title('Brain Tumor Detection using MRI Images')
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSo1iF3NWNq9053XsJGPrzFRsVgFF-_LhzJNA&s',width=500)
    app_mode = st.sidebar.selectbox("Select Page",["Home","Disease Identification"])

    #Main Page
    if(app_mode=="Home"):
        # image_path = "home_page.jpeg"
        # st.image(image_path,use_column_width=True)
        st.markdown("""
        This application is a deep learning-powered tool designed to assist in the classification of brain MRI images. It enables quick and automated detection of common brain tumor types, potentially aiding early diagnosis and treatment planning. The model is trained on labeled MRI scans and distinguishes among four key categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

⚠️ This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.

🧾 Tumor Class Descriptions
1. Glioma Tumor
Description: Gliomas are tumors that arise from glial cells in the brain or spinal cord. They are among the most common and aggressive types of brain tumors.

Characteristics:

Can occur in the cerebrum, cerebellum, or spinal cord.

Symptoms include headaches, seizures, and cognitive or personality changes.

Vary from low-grade (slow growing) to high-grade (fast growing and malignant).

2. Meningioma Tumor
Description: Meningiomas originate in the meninges, the protective membranes covering the brain and spinal cord. They are typically benign but can still cause significant problems due to pressure on adjacent brain tissue.

Characteristics:

Often found incidentally during imaging.

Common in middle-aged and older adults.

May lead to vision problems, headaches, or neurological deficits depending on size and location.

3. Pituitary Tumor
Description: These tumors grow in the pituitary gland, which regulates vital hormones. Most pituitary tumors are benign and known as adenomas.

Characteristics:

May cause hormonal imbalances.

Symptoms include vision problems, fatigue, unexplained weight changes, and menstrual disturbances.

Usually located at the base of the brain near the optic chiasm.

4. No Tumor
Description: MRI images in this category show normal brain structures without any detectable tumor. This class helps ensure the model doesn't overpredict abnormalities when none exist.

Characteristics:

Useful as a control category in classification models.

Indicates a healthy brain scan or no tumor presence detectable via imaging.""")


    elif(app_mode=="Disease Identification"):
        IMG_SIZE = (150, 150)
        class_labels = ['glioma', 'meningioma', 'notumor','pituitary']
        st.write("Upload an image to classify it")
    
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess
            img = img.resize(IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

    # Predict
            predictions = brain_tumor_model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]
            confidence = predictions[predicted_index]

            st.markdown(f"###  Predicted Class: `{predicted_label}`")
            st.markdown(f"### Confidence: `{confidence:.2f}`")

    # Optional: show all scores
            st.subheader("Class Probabilities:")
            for idx, score in enumerate(predictions):
                st.write(f"{class_labels[idx]}: {score:.4f}")

if selected == 'Eye Disease Prediction using OCT Images':
    st.title('Eye Disease Prediction using OCT Images')
    st.image('https://www.news-medical.net/images/news/ImageForNews_759142_16950000382822104.jpg',width=500)
    app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Identification"])

    #Main Page
    if(app_mode=="Home"):
        # image_path = "home_page.jpeg"
        # st.image(image_path,use_column_width=True)
        st.markdown("""
        ## **OCT Retinal Analysis Platform**

    #### **Welcome to the Retinal OCT Analysis Platform**

    **Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

    ##### **Why OCT Matters**
    OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

    ---

    #### **Key Features of the Platform**

    - **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
    - **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
    - **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

    ---

    #### **Understanding Retinal Diseases through OCT**

    1. **Choroidal Neovascularization (CNV)**
       - Neovascular membrane with subretinal fluid
       
    2. **Diabetic Macular Edema (DME)**
       - Retinal thickening with intraretinal fluid
       
    3. **Drusen (Early AMD)**
       - Presence of multiple drusen deposits

    4. **Normal Retina**
       - Preserved foveal contour, absence of fluid or edema

    ---

    #### **About the Dataset**

    Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
    - **Normal**
    - **CNV**
    - **DME**
    - **Drusen**

    Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

    ---

    #### **Get Started**

    - **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
    - **Explore Results**: View categorized scans and detailed diagnostic insights.
        """)

    #About Project
    elif(app_mode=="About"):
        st.header("About")
        st.markdown("""
                    #### About Dataset
                    Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. 
                    Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.
                    (A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). 
                    (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). 
                    (Middle right) Multiple drusen (arrowheads) present in early AMD. 
                    (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

                    ---

                    #### Content
                    The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). 
                    There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

                    Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

                    Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

                    Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing exper- tise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset selection and stratification process is displayed in a CONSORT-style diagram in Figure 2B. To account for human error in grading, a validation subset of 993 scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.

                    """)

    #Prediction Page
    elif(app_mode=="Disease Identification"):
    
        #st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSo1iF3NWNq9053XsJGPrzFRsVgFF-_LhzJNA&s',width=500)
        IMG_SIZE = (150, 150)
        class_labels = ['CNV','NORMAL','DRUSEN','DME']
        st.write("Upload an image to classify it")
    
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess
            img = img.resize(IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

    # Predict
            predictions = eye_disease_model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]
            confidence = predictions[predicted_index]

            st.markdown(f"###  Predicted Class: `{predicted_label}`")
            st.markdown(f"### Confidence: `{confidence:.2f}`")

            st.subheader("Class Probabilities:")
            for idx, score in enumerate(predictions):
                st.write(f"{class_labels[idx]}: {score:.4f}")
