# 💳 Financial Transaction Risk Scoring System

An end-to-end Machine Learning application for detecting potentially fraudulent financial transactions and generating real-time risk scores through an interactive Streamlit web interface.

This project demonstrates the integration of data preprocessing, imbalance handling, model optimization, and deployment into a production-style fraud detection system.

## 📌 Overview

Fraud detection systems must identify abnormal transaction behavior while minimizing missed fraud cases. This project simulates a practical fraud monitoring system by combining:

- Supervised machine learning (Random Forest)

- Class imbalance handling (SMOTE)

- Decision threshold optimization

- Business-oriented risk scoring

- Real-time deployment using Streamlit

The focus extends beyond model training to include usability, interpretability, and deployability.


## ⚙️ Project Pipeline
### 1. Data Preprocessing

- One-hot encoding for categorical features

- Train-test split

- Handling class imbalance using SMOTE

### 2. Model Training

- Logistic Regression (baseline)

- Random Forest Classifier (selected model)

- Random Forest was chosen based on stronger fraud recall performance.

### 3. Threshold Optimization

- Instead of using the default 0.5 cutoff, the decision threshold was tuned to: Threshold = 0.3

This improves sensitivity to fraudulent transactions.

### 4. Risk Scoring

- Fraud probability is transformed into a business score:

Risk Score = Probability × 100

This makes model output easier to interpret in operational environments.


## Results

The Random Forest classifier demonstrated strong performance in identifying fraudulent behavior after addressing dataset imbalance and applying threshold tuning.

Rather than maximizing raw accuracy, the model was optimized to prioritize fraud detection sensitivity. This ensures that high-risk transactions are less likely to be overlooked.

The deployed system provides:

- Real-time fraud probability estimation

- Business-aligned risk scoring (0–100 scale)

- Clear decision logic (Fraud Alert / Normal Transaction)

- Contextual explanation for flagged transactions

The final application bridges model performance with practical usability.


## 🛠 Tech Stack

- Python

- Pandas

- NumPy

- Scikit-Learn

- Imbalanced-Learn (SMOTE)

- Streamlit


## Usage

To run the application locally, follow these steps:

1.Clone the repository or download the project files to your system.

2.Create a virtual environment (recommended) and install the required dependencies listed in requirements.txt.

3.Ensure that the trained model file (fraud_rf_model.pkl) is present inside the model/ directory.

4.Launch the application using:

  streamlit run app.py

5.Open the application in your browser and enter transaction details to evaluate fraud risk in real time.

Live Demo Link: https://fraud-risk-scoring-system-dwqzche6zkgvh99j9a7oaz.streamlit.app
