# 🛡️ PhishGuard – Phishing Website Detection Using Ensemble Learning

## 1. Project Overview

PhishGuard is a Machine Learning-based website classification system that analyzes extracted numerical website features to identify potentially suspicious websites.

The project uses multiple Machine Learning algorithms, compares their performance, and selects the best-performing model using evaluation metrics.

The selected model, Hist Gradient Boosting, is deployed through a FastAPI backend with an interactive web dashboard.

**Project Type:** Binary Classification  
**Domain:** Artificial Intelligence & Machine Learning  
**Best Model:** Hist Gradient Boosting

## 2. Problem Statement

Phishing websites imitate legitimate websites to trick users into sharing sensitive information.

PhishGuard aims to classify website records using extracted numerical features and Machine Learning techniques.

The system provides a prediction, confidence score, and feature importance analysis.

## 3. Objectives

- Analyze numerical website features.
- Perform data preprocessing and exploratory analysis.
- Train and compare multiple classification algorithms.
- Evaluate models using standard performance metrics.
- Deploy the selected model using FastAPI.
- Build an interactive prediction dashboard.
- Analyze important features using Permutation Importance.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| MLflow | Experiment tracking |
| FastAPI | Backend API |
| Pydantic | Input validation |
| HTML, CSS, JavaScript | Frontend |
| Pytest | Testing |
| Docker | Containerization |

## 5. Dataset Description

**Dataset:** `data/final_dataset.csv`

| Attribute | Description |
|---|---|
| Total records | 579,920 |
| Total columns | 76 |
| Input features | 74 numerical features |
| URL column | Original website URL |
| Target column | `label` |
| Missing values | 0 in initial inspection |

The original URL column is excluded from model training. The target column is used for binary classification.

### Data Contract

| Field | Data Type | Purpose |
|---|---|---|
| `url` | String | Original website URL; excluded from training |
| 74 feature columns | Numeric | Extracted website characteristics |
| `label` | Binary integer | Classification target (0 or 1) |

The 74 numerical features represent website and URL characteristics, such as hostname length, domain age, HTTPS usage, URL depth, and DNS-related information.

Feature-specific valid ranges should be determined from the original dataset documentation and feature extraction rules.

**Dataset source:** Add the original dataset URL here.

**Note:** Confirm the meaning of label 0 and label 1 using the original dataset documentation before interpreting predictions as legitimate or phishing.

## 6. Data Preprocessing

The preprocessing and model pipeline includes:

1. Separating input features and target labels.
2. Excluding the original URL column from training.
3. Checking missing values.
4. Checking duplicate records.
5. Checking infinite values.
6. Inspecting constant features and feature distributions.
7. Examining outliers and numerical feature characteristics.
8. Applying median imputation.
9. Applying Variance Threshold for feature selection.
10. Training and evaluating classification models.

The trained Hist Gradient Boosting pipeline uses median imputation and Variance Threshold before classification.

## 7. Machine Learning Algorithms

The project compares the following 10 algorithms:

1. Decision Tree
2. Random Forest
3. Extra Trees
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Gaussian Naive Bayes
7. AdaBoost
8. Gradient Boosting
9. Hist Gradient Boosting
10. Linear Support Vector Machine (SVM)

The best model is selected based on F1-score.

## 8. Model Evaluation

The selected model is **Hist Gradient Boosting**.

Evaluation results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 95.925% |
| Precision | 96.1207% |
| Recall | 93.9795% |
| F1-score | 95.0381% |
| ROC-AUC | 99.2748% |

**Important:** These metrics were obtained from the held-out test evaluation after training on a 20,000-row sample. They should not be interpreted as evaluation results on the entire 579,920-row dataset.

## 9. Feature Importance Analysis

PhishGuard uses **Permutation Importance** to identify which features have the greatest influence on model performance.

The analysis uses a balanced sample of 1,000 records, with 500 records from each class.

### Top 10 Important Features

| Rank | Feature |
|---:|---|
| 1 | hostname_length |
| 2 | domain_age_days |
| 3 | has_www |
| 4 | path_length |
| 5 | uses_https |
| 6 | ttl_value |
| 7 | uppercase_ratio |
| 8 | has_mx_record |
| 9 | url_depth |
| 10 | num_dots |

The importance score represents the decrease in F1-score when a feature is randomly shuffled. A higher score indicates greater importance to the model's predictive performance.

Results are saved in:

- `results/feature_importance.csv`
- `results/feature_importance.png`

## 10. Project Structure

```text
Phishing-Website-Detection/
│
├── data/
│   └── final_dataset.csv
│
├── model/
│   └── model.joblib
│
├── results/
│   ├── feature_importance.csv
│   └── feature_importance.png
│
├── static/
│   └── index.html
│
├── tests/
│   └── test_app.py
│
├── app.py
├── train.py
├── feature_importance.py
├── requirements.txt
├── requirements-serve.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## 11. Installation and Setup

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Phishing-Website-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```cmd
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 12. Run the Application

Start the FastAPI server:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open the dashboard:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

## 13. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Opens the web dashboard |
| GET | `/health` | Checks API health |
| GET | `/sample` | Returns a sample feature record |
| POST | `/predict` | Generates a prediction |
| GET | `/feature-importance` | Returns top 10 important features |

### Prediction API

The prediction endpoint accepts extracted numerical features.

Example request structure:

```json
{
  "features": {
    "feature_name": 0
  }
}
```

The complete request must contain all required feature names and valid numerical values.

Example response:

```json
{
  "prediction": 1,
  "predicted_label": 1,
  "result": "Class 1",
  "confidence": 99.03,
  "model": "Hist Gradient Boosting",
  "message": "Prediction generated successfully"
}
```

## 14. Testing

The project uses Pytest for automated testing.

Run:

```bash
python -m pytest tests/test_app.py -v
```

The following six test cases have been implemented:

1. Health endpoint test
2. Homepage loading test
3. Sample endpoint test
4. Valid prediction test
5. Invalid prediction test
6. Feature importance endpoint test

**Current testing status:** 6/6 tests passed.

## 15. MLflow Experiment Tracking

MLflow is used for tracking model configurations and evaluation metrics.

To start the MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

`http://127.0.0.1:5000`

## 16. Docker Deployment

Build the Docker image:

```bash
docker build -t phishing-detection .
```

Run the container:

```bash
docker run -p 8000:8000 --restart unless-stopped phishing-detection
```

Open:

`http://127.0.0.1:8000`

## 17. Limitations

- The current application expects pre-extracted numerical website features.
- It does not automatically extract all 74 features from a raw URL.
- Prediction quality depends on the quality and distribution of the input features.
- Dataset label interpretation must follow the original dataset documentation.
- Model evaluation metrics are based on the specified training and test sample, not the entire dataset.

## 18. Future Scope

- Automatic URL feature extraction.
- Real-time website analysis.
- Explainable AI integration.
- Model drift monitoring.
- Cloud-based deployment.
- Improved security and validation mechanisms.

## 19. Conclusion

PhishGuard demonstrates an end-to-end Machine Learning workflow covering data preprocessing, algorithm comparison, model evaluation, model deployment, API development, automated testing, and feature importance analysis.

The project combines Machine Learning with web technologies to create an interactive website classification system.

---

**Project:** PhishGuard – Phishing Website Detection Using Ensemble Learning

**Developed using:** Python, Scikit-learn, FastAPI, and HTML/CSS/JavaScript.

**Team Members:** Add team member names and contributions here.
