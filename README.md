# Machine Learning Assignment 2 - Traffic Violations

## Problem Statement

The objective of my project is to build and evaluate multiple machine learning classification models on a real-world dataset and deploy the results through an interactive web application.

Using the **Traffic Violations** data, this project implements six (6) different classifications models and compares their performance using standard evaluation metrics.
The whole project is then deployed as a Streamlit application for interactive model select, test data uploads and evaluation.

---

## About the Dataset

The dataset used in this project is a publically available Traffic Violations datasetsource from an open source data repository.

The dataset contains records of traffic violations along with driver, vehicle, location, and violation related attributes.

> ### Characteristics/Features

* **Source** - [Kaggle](https://www.kaggle.com/)
* **Link to Dataset** - [Indian Traffic Violations Dataset](https://www.kaggle.com/datasets/khushikyad001/indian-traffic-violation/data) from [Khushi Yadav](https://www.kaggle.com/khushikyad001)
* **Type** - Multi-class classification dataset
* **Number of Instances** - More than 500
* **Number of Features** - More Than 12 (post preprocessing dataset)
* **Data Types** - Numerical and Categorical

> **Violation Types (classes):**

* `Drunk Driving`, `Driving without License`, `No Helmet`, `No Seatbelt`, `Signal Jumping`, `Wrong Parking`, `Overspeeding`, `Overloading`, `Other`

## Machine Learning Models

The following classification models are implemented using the same dataset:

* Logistic Regression
* Decision Tree Classifier
* K-Neared Neighbor (kNN)
* Naive Bayes Classifier
* Random Forest Classifier (Ensemble)
* XGBoost Classifier (Ensemble)

---

## Model Evaluation Metrics

Each model is evaluated using the following metrics:

* Accuracy
* Area Under the Curve (AUC)
* Precision
* Recall
* F-1 Score
* Mathews Correlation Coefficient (MCC)

## Performance Comparison Table

To be updated later.

| Model Name            | Accuracy | AUC | Precision | Recall | F-1 Score | MCC |
|-----------------------|----------|-----|-----------|--------|-----------|-----|
| Logistic Regression   |          |     |           |        |           |     |
| Decision Tree         |          |     |           |        |           |     |
| kNN                   |          |     |           |        |           |     |
| Naive Bayes           |          |     |           |        |           |     |
| Random Forest         |          |     |           |        |           |     |
| XGBoost               |          |     |           |        |           |     |

---

## Model Performance Observations

| Model Name                      |  Observations                               |
|---------------------------------|---------------------------------------------|
| Logistic Regression             |                                             |
| Decision Tree                   |                                             |
| kNN                             |                                             |
| Naive Bayes                     |                                             |
| Random Forest                   |                                             |
| XGBoost                         |                                             |

---

## Streamlit Application Features

The Streamlit web application provides the following features

* Upload Test dataset in CSV format
* Select Classification model from a dropdown
* Display evaluation metrics
* Show Confusion Matrix or classification Repotrs

---

## Technologies/Libraries Used

* Python
* scikit-learn
* XGBoost
* Pandas
* Numpy
* Streamlit
* Matplotlib / Seaborn

---

## Author

Tejaswee Gaur

---

## License

This project is for educational purposes as part of BITS Pilani ML Assignment 2.

---

## Acknowledgements

* Kaggle [Indian Traffic Violations Dataset](https://www.kaggle.com/datasets/khushikyad001/indian-traffic-violation/data) from [Khushi Yadav](https://www.kaggle.com/khushikyad001)
* BITS Pilani for the assignment framework (S1-25_AIMLCZG565).
