# Machine Learning Assignment 2 – Traffic Violations

## Problem Statement

The objective of this project is to build, evaluate, and compare multiple machine learning classification models on a real-world dataset and deploy the results through an interactive web application.

Using the **Traffic Violations** dataset, this project implements six (6) different classification models and compares their performance using standard multi-class evaluation metrics.  
The complete workflow — from data preparation to model evaluation — is designed to follow best practices and is finally deployed as a **Streamlit application** for interactive model selection, test data upload, and performance visualization.

---

## Dataset Description

The dataset used in this project is a publicly available **Indian Traffic Violations** dataset sourced from Kaggle.

The dataset contains records of traffic violations along with driver, vehicle, location, and violation-related attributes.

### Characteristics / Features

- **Source**: Kaggle  
- **Dataset Link**: [Indian Traffic Violations Dataset](https://www.kaggle.com/datasets/khushikyad001/indian-traffic-violation/data)  
- **Type**: Multi-class classification  
- **Number of Instances**: ~4,000  
- **Number of Features**: 24 (after preprocessing)  
- **Data Types**: Numerical and Categorical  

### Violation Types (Target Classes)

- Drunk Driving  
- Driving Without License  
- No Helmet  
- No Seatbelt  
- Over-speeding  
- Overloading  
- Signal Jumping  
- Using Mobile Phone  
- Wrong Parking  

---

## Models Used

The following classification models were implemented and evaluated using the same preprocessing pipeline and train–test split:

- Logistic Regression  
- k-Nearest Neighbors (kNN)  
- Decision Tree Classifier  
- Naive Bayes Classifier  
- Random Forest Classifier (Ensemble)  
- XGBoost Classifier (Ensemble)  

---

## Model Evaluation Metrics

Each model was evaluated on a held-out test set using the following metrics:

- Accuracy  
- Precision (Macro Averaged)  
- Recall (Macro Averaged)  
- F1-Score (Macro Averaged)  
- Area Under the Curve (AUC – One-vs-Rest)  
- Matthews Correlation Coefficient (MCC)  

Additionally, the **training time** and **evaluation (prediction) time** were recorded for each model to compare computational efficiency.

> **Note:** Timing results are machine-dependent and are included for *relative comparison only*.

---

## Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | AUC (OvR) | MCC | Training Time (s) | Evaluation Time (s) |
|------|----------|-----------|--------|----------|-----------|-----|------------------|---------------------|
| Logistic Regression | 0.1200 | 0.1190 | 0.1185 | 0.1179 | 0.5028 | 0.0087 | 0.10 | 0.02 |
| kNN | 0.1250 | 0.1209 | 0.1228 | 0.1147 | 0.4928 | 0.0134 | 0.03 | 2.07 |
| Decision Tree | 0.1238 | 0.1238 | 0.1232 | 0.1232 | 0.5068 | 0.0141 | 0.09 | 0.01 |
| Random Forest | 0.1088 | 0.1037 | 0.1060 | 0.1023 | 0.4861 | −0.0052 | 1.61 | 0.13 |
| Naive Bayes | **0.1338** | **0.1327** | **0.1330** | **0.1323** | 0.5054 | **0.0251** | 0.05 | 0.05 |
| XGBoost | 0.1225 | 0.1207 | 0.1209 | 0.1196 | 0.5068 | 0.0111 | 0.81 | 0.14 |

---

## Model Performance Observations

### Overall Observations

- Model performance across all classifiers remains close to random guessing (~11–13% accuracy for 9 classes), indicating **weak feature–target correlation**.
- The task of predicting the exact traffic violation type is inherently challenging due to overlapping class characteristics and limited discriminatory power in the available features.
- Increasing model complexity does not necessarily result in improved performance for this dataset.

---

### Model-wise Observations

| Model | Observations |
|------|--------------|
| Logistic Regression | Serves as a strong baseline with very fast training and inference, but limited by linear decision boundaries in a complex multi-class setting. |
| kNN | Slight improvement over Logistic Regression but suffers from extremely high prediction time, making it unsuitable for real-time inference. |
| Decision Tree | Captures non-linear relationships better than linear models but remains sensitive to noise in categorical features. |
| Naive Bayes | Achieved the best overall performance (highest macro F1-score and MCC), indicating that probabilistic modeling is effective for this dataset. |
| Random Forest | Underperformed despite higher computational cost, likely due to noisy one-hot encoded categorical features. |
| XGBoost | Did not significantly outperform simpler models, suggesting limited predictive signal in the dataset despite its advanced ensemble structure. |

---

## Confusion Matrices and Per-Class Metrics

For each model:

- A **confusion matrix** was generated and saved as a CSV file.
- **Per-class precision, recall, F1-score, and support** were computed using `classification_report`.

These artifacts are stored under:
```text
| models/artifacts/
├── confusion_matrices/
├── classification_reports/
```

They are also used for visualization in the Streamlit application.

---

## Streamlit Application Features

The Streamlit web application provides the following features:

- Upload a test dataset in CSV format  
- Select a classification model from a dropdown  
- Display overall evaluation metrics  
- Visualize confusion matrices  
- View per-class performance metrics  

---

## Technologies / Libraries Used

- Python  
- scikit-learn  
- XGBoost  
- Pandas  
- NumPy  
- Streamlit  
- Matplotlib / Seaborn  

---

## Author

**Tejaswee Gaur**

---

## License

This project is for educational purposes as part of **BITS Pilani – Machine Learning Assignment 2**.

---

## Acknowledgements

- Kaggle – [Indian Traffic Violations Dataset](https://www.kaggle.com/datasets/khushikyad001/indian-traffic-violation/data) by Khushi Yadav  
- BITS Pilani – Assignment Framework (S1-25_AIMLCZG565)
