# README  

# 💳 Fraudulent Transaction Detection using Machine Learning  

## 🔍 Overview  
This project builds a **machine learning model** to detect **fraudulent transactions** using transaction data. With over **650,000 transactions** analyzed, our model aims to protect financial institutions by identifying anomalies in financial flows effectively and efficiently.  

The dataset includes essential features like transaction type, amounts, balances, and fraud flags, enabling us to build a robust pipeline for prediction. This project uses **cutting-edge machine learning algorithms** to achieve high accuracy and performance in fraud detection.  

---

## 📁 Dataset Description  
The dataset contains the following features:  

| **Feature**         | **Description**                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| `step`              | Time step of the transaction.                                                  |
| `type`              | Transaction type: PAYMENT, TRANSFER, CASH_OUT, etc.                            |
| `amount`            | Amount of the transaction.                                                     |
| `nameOrig`          | Identifier of the sender.                                                      |
| `oldbalanceOrg`     | Initial balance of the sender before the transaction.                          |
| `newbalanceOrig`    | Balance of the sender after the transaction.                                   |
| `nameDest`          | Identifier of the recipient.                                                   |
| `oldbalanceDest`    | Initial balance of the recipient before the transaction.                       |
| `newbalanceDest`    | Balance of the recipient after the transaction.                                |
| `isFraud`           | Indicates if the transaction was fraudulent (1: Fraudulent, 0: Legitimate).    |
| `isFlaggedFraud`    | Flags suspicious transactions (1: Flagged, 0: Not Flagged).                    |  

---

## 🚀 Models Used  

This project evaluates several regression and classification models for detecting fraud.  

### Regression Models:
- **Linear Regression**  
- **Decision Tree Regression**  
- **Random Forest Regression**  
- **Extra Trees Regression**  

### Classification Models:  
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **Gradient Boosting Classifier**  
- **AdaBoost Classifier**  
- **Extra Trees Classifier**  
- **K-Nearest Neighbors (KNN)**  
- **Support Vector Classifier (SVC)**  
- **Naive Bayes (GaussianNB)**  

---

## 📊 Results  

### **Regression Performance:**  
| **Model**                  | **Mean Squared Error (MSE)**         |  
|----------------------------|--------------------------------------|  
| Linear Regression          | 0.0303                              |  
| Decision Tree Regression   | 0.0349                              |  
| Random Forest Regression   | 0.0277                              |  
| Extra Tree Regression      | 0.0290                              |  

### **Classification Performance:**  
- **Accuracy:** 99.91%  
- **Confusion Matrix:**  
  ```
  [[649645     30]
   [   579     28]]
  ```  
- **Precision, Recall, F1-score:**  
  ```
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    649675
           1       0.48      0.05      0.08       607

    accuracy                           1.00    650282
   macro avg       0.74      0.52      0.54    650282
weighted avg       1.00      1.00      1.00    650282
  ```  

- **Key Insights:**  
  - The model achieved **exceptionally high accuracy** for legitimate transactions.
  - While precision for fraud detection is promising, recall needs improvement for better fraud identification.  

---

## 🛠️ Installation  

### Prerequisites:  
- Python 3.8+  
- Libraries:  
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn
  ```  

### Run the Project:  
1. Clone the repository:  
   ```bash
   git clone https://github.com/<username>/fraud-detection.git
   ```  
2. Navigate to the project directory:  
   ```bash
   cd fraud-detection
   ```  
3. Run the script to train and evaluate the model:  
   ```bash
   python fraud_detection.py
   ```  

---

## 📈 Future Improvements  

- Implement advanced techniques like **Deep Learning** for feature extraction and prediction.  
- Fine-tune hyperparameters of the models to improve recall for fraudulent transactions.  
- Explore ensemble techniques for better prediction accuracy.  
- Build a real-time fraud detection pipeline using **Apache Kafka** or **Spark Streaming**.  

---

## 🤝 Contributions  

Contributions are welcome! If you have ideas to improve the detection algorithm, please:  
1. Fork the repo.  
2. Create a feature branch.  
3. Submit a pull request.  

---

## 📜 License  

This project is licensed under the MIT License.  

---

### 🌟 If you like this project, give it a star ⭐ and let’s build safer financial systems together!
