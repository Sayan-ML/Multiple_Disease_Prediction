# 🧠 Multiple Disease Prediction System

An integrated **Machine Learning** and **Deep Learning** based web application that predicts multiple diseases, including:  

- ❤️ **Heart Disease**  
- 🩺 **Diabetes**  
- 🧪 **Kidney Disease**  
- 🌫️ **Pneumonia (via Chest X-rays)**  
- 👁️ **Diabetic Retinopathy (via Retinal Images)**  

Built using **Scikit-learn**, **TensorFlow/Keras**, and **Streamlit**, this system provides a **one-stop solution** for early disease detection through both **clinical parameters** and **medical imaging**.

---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)  
![Keras](https://img.shields.io/badge/Keras-DL-red?logo=keras&logoColor=white)  
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-yellow)  

---

## 📖 Project Overview  

The **Multiple Disease Prediction System** is an integrated **Machine Learning (ML) and Deep Learning (DL)** powered **web application** designed to predict multiple diseases from **clinical parameters and medical imaging**.  

This platform acts as a **one-stop solution for early disease detection**, helping healthcare professionals, researchers, and patients make **data-driven decisions**. By combining structured medical data (blood tests, vitals, clinical parameters) with unstructured medical imaging (X-rays, retinal scans), the system achieves both **breadth and depth** in diagnosis.  

The application is implemented using **Scikit-learn, TensorFlow/Keras, and Streamlit**, ensuring an intuitive, interactive, and user-friendly interface for both technical and non-technical users.  

With real-time predictions, clear result visualization, and multi-disease support, this tool aims to bridge the gap between **clinical diagnostics and AI-driven healthcare innovations**.  

---

## 🚀 Live Demo  

📽️ [App Walkthrough Demo](https://drive.google.com/file/d/1AdHwqrB8fRaRpxSjY3AOvnAwjWiRIX--/view?usp=sharing)  

---

## ✨ Features  

- ✅ **Multi-disease support** (Heart, Diabetes, Kidney, Pneumonia, Retinopathy)  
- ✅ **Tab-based navigation** for a smooth user experience  
- ✅ **Interactive forms** for entering medical parameters  
- ✅ **CNN-based medical image classification** for Pneumonia & Retinopathy  
- ✅ **Fast & optimized predictions** using pre-trained models  
- ✅ **Optional confidence scores** for transparency in predictions  
- ✅ **Responsive UI** built with **Streamlit**  
- ✅ **Scalable architecture** – easily extendable to new diseases  

---

## 🧬 Diseases Covered  

### ❤️ Heart Disease Prediction  
- **Inputs:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, etc.  
- **Model:** Random Forest / Logistic Regression  
- **Goal:** Early detection of cardiovascular disease risk.  

### 🩺 Diabetes Prediction  
- **Inputs:** Glucose, Blood Pressure, BMI, Insulin, Age, Skin Thickness, etc.  
- **Model:** Support Vector Machine / Decision Tree  
- **Goal:** Detect likelihood of Type 2 Diabetes for early lifestyle/medical interventions.  

### 🧪 Kidney Disease Prediction  
- **Inputs:** Serum Creatinine, Sodium, Potassium, Albumin, Blood Urea, etc.  
- **Model:** Random Forest Classifier  
- **Goal:** Identify potential Chronic Kidney Disease (CKD) cases.  

### 🌫️ Pneumonia Detection (via Chest X-rays)  
- **Input:** Chest X-ray image (PNG/JPG)  
- **Model:** Deep **Convolutional Neural Network (CNN)**  
- **Dataset:** Kaggle Chest X-Ray dataset  
- **Goal:** Detect lung infections and classify images as **Pneumonia / Normal**.  

### 👁️ Diabetic Retinopathy Detection (via Retinal Images)  
- **Input:** Retinal Fundus Image  
- **Model:** Deep CNN  
- **Dataset:** APTOS 2019 / EyePACS (Kaggle)  
- **Goal:** Detect retinal damage due to prolonged diabetes.  

---

## 🛠️ Tech Stack  

| Category       | Tools/Frameworks |
|----------------|------------------|
| **Frontend**   | Streamlit |
| **ML Models**  | Scikit-learn |
| **DL Models**  | TensorFlow, Keras |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit Cloud / Local Server |

---

## 📁 Project Structure  

```bash
├── app.py                      # Main Streamlit app
├── heart_disease_model.pkl     # Saved ML model
├── diabetes_model.pkl
├── kidney_model.pkl
├── pneumonia_model.h5          # CNN model for Pneumonia
├── retinopathy_model.h5        # CNN model for Retinopathy
├── utils/                      # Preprocessing & prediction logic
│   ├── preprocess.py
│   ├── prediction.py
│   └── image_utils.py
├── data/                       # Sample data or dataset references
├── requirements.txt
└── README.md

├── requirements.txt
└── README.md
```

---

## 🌟 Future Improvements  

- 🔹 Add more diseases (e.g., **Parkinson’s, Liver Disease, Alzheimer’s**)  
- 🔹 Integrate **Wearable IoT device data** (Fitbit, Apple Watch)  
- 🔹 Add **Explainable AI (XAI)** for transparent predictions  
- 🔹 Deploy with **Docker + Kubernetes** for production scalability  
- 🔹 Add **mobile app support** (Flutter/React Native)  

---

## 🤝 Contributing  

We welcome contributions from the community! 🎉  

1. **Fork** the repository  
2. **Create a new branch** (`feature-xyz`)  
3. **Make your changes**  
4. **Commit and push** your branch  
5. **Open a Pull Request** 🚀  

💡 You can also open issues for **bug reports, feature requests, or documentation improvements**.  

---

## 💖 Show Your Support  

If you find this project useful, please **give it a ⭐ on GitHub** to help others discover it!  

---

## 📜 License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

## 👨‍💻 Developed By Sayan Banerjee

<p align="left">
  <a href="https://www.linkedin.com/in/sayan-banerjee-0222a4214/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" alt="LinkedIn Logo"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/Sayan-ML" target="_blank">
    <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" width="30" alt="GitHub Logo"/>
  </a>
</p>

