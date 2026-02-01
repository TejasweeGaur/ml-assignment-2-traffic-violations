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

### [Click here to view detailed Feature Analysis](datasets/README.md)

---

## Models Used

The following classification models were implemented and evaluated using the same preprocessing pipeline and train–test split:

- Logistic Regression  
- k-Nearest Neighbors (kNN)  
- Decision Tree Classifier  
- Naive Bayes Classifier  
- Random Forest Classifier (Ensemble)  
- XGBoost Classifier (Ensemble)  

### [Click here to view detailed Model Analysis and Processing Workflow](models/README.md)

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

## Key Insights

- Predicting exact traffic violation types is inherently challenging due to weak feature–target correlation.
- Simpler probabilistic models (Naive Bayes) performed competitively against complex ensemble models.
- Model complexity does not guarantee better performance in noisy, multi-class datasets.
- Class-wise evaluation and balanced metrics are essential for fair assessment.

---

## Reproducibility and Assumptions

- All models were trained using the same preprocessing pipeline and data split.
- Random seeds were fixed where applicable.
- Performance metrics may vary depending on environment, hardware, and library versions.
- Results are intended for comparative analysis rather than absolute performance benchmarks.

---

## Confusion Matrices and Per-Class Metrics

For each model:

- A **confusion matrix** was generated and saved as a CSV file.
- **Per-class precision, recall, F1-score, and support** were computed using `classification_report`.

These artifacts are stored under:
```text
models/artifacts/
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

## Security Considerations

### Model File Deserialization

This application uses `joblib.load()` to load pre-trained machine learning models stored as pickle files. 

**⚠️ Security Risk**: Pickle deserialization can execute arbitrary Python code. This poses a security risk if pickle files from untrusted sources are loaded.

**Current Mitigation**: 
- All model files (`.pkl`) are committed directly to this repository and are trusted.
- The application only loads models from the `models/artifacts/` directory.
- Users do not have the ability to upload or specify custom model files.

**Future Considerations**:
- If this application is extended to allow users to upload custom model files, implement strict security controls:
  - Use sandboxed environments or containers
  - Implement file validation and scanning
  - Consider alternative serialization formats (e.g., ONNX, SafeTensors)
  - Add explicit user warnings about security risks
  - Restrict file sources to trusted origins only

For more information on pickle security, see: [Python Pickle Documentation](https://docs.python.org/3/library/pickle.html#module-pickle)

---

## Project Structure

``` markdown
ml-assignment-2-traffic-violations/
│
├── datasets/
│   ├── __init__.py
│   ├── setup_dataset.py
│   ├── dataset.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── data_analysis.py
│   ├── data_preparation.py
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── artifacts/
│   │   ├── *.pkl
│   │   ├── label_encoder.pkl
│   │   ├── model_comparison.csv
│   │   ├── confusion_matrices/
│   │   └── classification_reports/
│
├── app.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── other files

```

---

## How to run the Project

Follow the steps below to run the project locally and explore the trained machine learning models through the Streamlit application.

---

### 1. Prerequisites

Ensure that you have the following installed:

- Python **3.9 or higher**
- `pip` (Python package manager)
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/TejasweeGaur/ml-assignment-2-traffic-violations.git -b main
cd ml-assignment-2-traffic-violations
```

### 3. Create and Activate a Virtual Environment (Recommended)

#### Windows
``` bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux
``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

Install all required Python libraries using:

``` python
pip install -r requirements.txt
```

### 5. Prepare the Dataset

Download and set up the dataset from Kaggle:

``` python
python -m datasets.setup_dataset
```
This step downloads the dataset and stores it as: ***datasets/dataset.csv***

### 6. Run Data Preparation (Train/Test Split)

Generate the training and testing datasets:
``` python
python -m models.data_preparation
```

This creates:
 * datasets/train.csv
 * datasets/test.csv

### 7. Train All Models and Generate Artifacts
Train all classification models, compute evaluation metrics, and save artifacts:
``` python
python -m models.train_models
```
This step generates:
 * Trained model pipelines (.pkl files)
 * Label encoder
 * Model comparison metrics
 * Confusion matrices
 * Per-class classification reports

All artifacts are saved under: ***models/artifacts/***

### 8. Launch the Streamlit Application
Start the Streamlit web interface:
 ``` bash
streamlit run streamlit_app/app.py
``` 
Once started, open the URL shown in the terminal (usually http://localhost:8501) in your browser.

### 9. Using the Streamlit App
Within the application, you can:

* Download a sample test CSV
* Upload a CSV file for evaluation
* Select a trained model dynamically
* View overall evaluation metrics
* Analyze confusion matrices (raw or normalized)
* Inspect per-class performance metrics
* Compare all models using tables and bar charts

> **Notes**
  Results may vary slightly depending on the system environment, library versions, and random seeds.
  The metrics shown in the Streamlit app correspond to the same training run documented in the project README.
  No model retraining occurs inside the Streamlit application; all models are loaded from saved artifacts.

---

## Technologies / Libraries Used

* [Python](https://www.python.org/)  
* [scikit-learn](https://scikit-learn.org/)  
* [XGBoost](https://xgboost.readthedocs.io/)  
* [Pandas](https://pandas.pydata.org/)  
* [NumPy](https://numpy.org/)  
* [Streamlit](https://streamlit.io/)  
* [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)  
* [Joblib](https://joblib.readthedocs.io/)  
* [Kaggle](https://www.kaggle.com/)  

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
