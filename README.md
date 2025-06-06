🩺 Diabetes Prediction System
A Machine Learning-based web application that predicts whether a person is likely to have diabetes or not, based on diagnostic measurements. Built using Python, Scikit-learn, and deployed using Streamlit.

🚀 Demo
Try the live app:

📌 Features
User-friendly web interface powered by Streamlit

Supports real-time diabetes risk prediction

Uses a trained machine learning model (e.g., Logistic Regression, Random Forest, etc.)

Clean visualizations and intuitive inputs (sliders, checkboxes)

Handles missing values and outliers appropriately

Optionally shows model performance metrics (Accuracy, Confusion Matrix, etc.)

📊 Input Features
The app uses the following health indicators for prediction:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI (Body Mass Index)

Diabetes Pedigree Function

Age

🛠️ Tech Stack
Frontend: Streamlit

Backend: Python, Scikit-learn

Visualization: Matplotlib, Seaborn

Model Persistence: Pickle

🧠 ML Model
A machine learning classification model is trained using the PIMA Indian Diabetes dataset from Kaggle.
It has been preprocessed with the following steps:

Outlier handling

Imputation for missing values

Feature scaling

Model training and evaluation


# Run the app
streamlit run app.py
📁 Repository Structure
bash
Copy
Edit
├── app.py                 # Streamlit app
├── model.pkl              # Trained ML model
├── diabetes.csv           # Dataset used
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
