# Customer Churn Prediction

This project predicts whether a customer will churn based on their details and service usage.
It includes data preprocessing, EDA, model training, and saving the trained model as a `.pkl` file.

---

## 📁 Project Structure

```
customer-churn/
│── processed.ipynb              # Data cleaning + preprocessing
│── customer_main.ipynb          # Model training + evaluation
│── churn_model.pkl              # Saved trained model
│── preprocessed_data.csv        # Cleaned dataset (optional)
│── requirements.txt             # Python dependencies
│── README.md                    # Project overview
```

---

## 🔍 What This Project Does

* Loads customer churn dataset
* Cleans and preprocesses the data
* Handles missing values and encodes categorical features
* Splits the data into train/test sets
* Trains a machine learning model (Logistic Regression / Random Forest)
* Evaluates model accuracy
* Saves the final model as **churn_model.pkl**
* Prepares the project for deployment (optional)

---

## 📊 Techniques Used

* Pandas for data cleaning
* NumPy for numerical operations
* Seaborn/Matplotlib for EDA
* Scikit-learn for ML models
* Joblib/Pickle for saving the model

---

## 🚀 How to Run This Project

1. Install the requirements:

```
pip install -r requirements.txt
```

2. Open the notebooks:

* `processed.ipynb` → preprocessing
* `customer_main.ipynb` → model training

3. The trained model will be saved as:

```
churn_model.pkl
```

---

## 📈 Model Output

The model predicts:

* **1 → Customer will churn**
* **0 → Customer will not churn**

---

## 🌐 Deployment (Optional)

If you want to deploy, create an `app.py` using Streamlit and load the `.pkl` model.

I can generate this for you if needed.

---

## 📝 Author

Created by Lalit Shinde
Machine Learning / AIML Engineering Student
