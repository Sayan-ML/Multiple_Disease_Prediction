🧠 Multiple Disease Prediction System
An integrated Machine Learning and Deep Learning based web application that predicts multiple diseases including:

❤️ Heart Disease

🩺 Diabetes

🧪 Kidney Disease

🌫️ Pneumonia (via Chest X-rays)

👁️ Diabetic Retinopathy (via Retinal Images)

Built using Scikit-learn, TensorFlow/Keras, and Streamlit, this system provides a one-stop solution for early disease detection through both clinical parameters and medical imaging.

🚀 Live Demo
👉 Launch the App
(Replace with actual Streamlit Share or hosted link)

📌 Features
✅ Tab-based navigation for each disease prediction
✅ Real-time predictions with user-friendly input interface
✅ Tabular models for Diabetes, Heart, and Kidney prediction
✅ CNN-based image classification for Pneumonia & Retinopathy
✅ Streamlined preprocessing and model inference
✅ Clear results with optional model confidence scores
✅ Beautiful, responsive UI using Streamlit

🧬 Diseases Covered
1. ❤️ Heart Disease Prediction
Inputs: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, etc.

Model: Random Forest / Logistic Regression

2. 🩺 Diabetes Prediction
Inputs: Glucose, Blood Pressure, BMI, Insulin, Age, etc.

Model: Support Vector Machine / Decision Tree

3. 🧪 Kidney Disease Prediction
Inputs: Serum Creatinine, Sodium, Potassium, Albumin, Blood Urea, etc.

Model: Random Forest Classifier

4. 🌫️ Pneumonia Detection
Input: Chest X-ray image (PNG/JPG)

Model: Convolutional Neural Network (CNN)

Dataset: Chest X-Ray Images (Kaggle)

5. 👁️ Diabetic Retinopathy Detection
Input: Retinal fundus image

Model: Deep CNN

Dataset: APTOS 2019 / EyePACS (Kaggle)

🛠️ Tech Stack
Category	Tools
Frontend	Streamlit
ML Models	Scikit-learn
DL Models	TensorFlow / Keras
Visualization	Matplotlib, Seaborn
Deployment	Streamlit Cloud / Local Server

📁 Project Structure
bash
Copy
Edit
├── app.py                      # Main Streamlit app
├── heart_disease_model.pkl     # Saved model
├── diabetes_model.pkl
├── kidney_model.pkl
├── pneumonia_model.h5          # CNN model
├── retinopathy_model.h5
├── utils/                      # Preprocessing & prediction logic
│   ├── preprocess.py
│   ├── prediction.py
│   └── image_utils.py
├── data/                       # Sample data or dataset references
├── requirements.txt
└── README.md

# Run the Streamlit app
streamlit run app.py

📊 Model Performance (Optional)
Disease	Model	Accuracy
Heart Disease	Random Forest	85%
Diabetes	SVM	78%
Kidney Disease	RF	90%
Pneumonia	CNN	94%
Retinopathy	CNN	92%

🙋‍♂️ Author
Sayan Banerjee
🎓 MSc in Statistics and Computing, BHU
💼 Skilled in Python, Machine Learning, Deep Learning, Power BI, SQL
📫 LinkedIn | Email

📄 License
This project is licensed under the MIT License.

App Demo :- https://drive.google.com/file/d/1AdHwqrB8fRaRpxSjY3AOvnAwjWiRIX--/view?usp=sharing
