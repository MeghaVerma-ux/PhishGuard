# PhishGuard – Phishing Website Detection Using Machine Learning

## Project Overview
PhishGuard is a machine-learning-based web application that detects potentially phishing websites by analysing URL characteristics, domain information, DNS records and WHOIS attributes.

Users can enter a URL and receive a prediction through a web-based interface.

## Objectives
- Detect potentially malicious URLs.
- Analyse URL and domain characteristics.
- Apply data preprocessing and machine learning.
- Evaluate model performance.
- Deploy the model through a FastAPI backend.
- Provide an interactive URL-scanning interface.

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI and Uvicorn
- HTML, CSS and JavaScript
- dnspython and python-whois
- Git and GitHub
- Render
- Docker

## Dataset
- Total records: 579,920
- Features: 74
- URL column: `url`
- Target column: `label`

Target distribution:
- Label 0: 339,074 records
- Label 1: 240,846 records

The class mapping is provisional and should be confirmed against the original dataset documentation.

## Data Preprocessing
The project includes:
- Dataset loading and inspection
- Missing-value analysis
- Duplicate-value checks
- Infinite-value checks
- Constant-feature analysis
- Target distribution analysis
- Feature selection and preparation

The deployed model pipeline uses:
1. SimpleImputer with median strategy
2. VarianceThreshold
3. HistGradientBoostingClassifier

## Machine Learning Model
The deployed model is Hist Gradient Boosting Classifier.

Configuration:
- max_iter = 60
- max_leaf_nodes = 25
- random_state = 42

The trained model is stored at `model/model.joblib`.

## Model Evaluation
Previously recorded held-out evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 95.925% |
| Precision | 96.1207% |
| Recall | 93.9795% |
| F1-score | 95.0381% |
| ROC-AUC | 99.2748% |

These are recorded evaluation results, not guaranteed real-world performance.

## Application Workflow
1. User enters a URL.
2. Frontend sends it to FastAPI.
3. Backend extracts URL features.
4. DNS and WHOIS information is collected where available.
5. Features are passed to the trained model.
6. Model generates a prediction.
7. Result is displayed on the frontend.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web application |
| `/health` | GET | Health check |
| `/sample` | GET | Sample information |
| `/feature-importance` | GET | Feature importance |
| `/predict` | POST | Prediction from features |
| `/scan` | POST | URL scanning |

## Local Setup

Clone the repository:

```bash
git clone https://github.com/MeghaVerma-ux/PhishGuard.git
cd PhishGuard
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements-serve.txt
```

Run the application:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Deployment

- GitHub: https://github.com/MeghaVerma-ux/PhishGuard
- Live application: https://phishguard-y1s9.onrender.com
- Health endpoint: https://phishguard-y1s9.onrender.com/health

## Current Limitations
- The scanner currently extracts 73 of the 74 training features.
- The original training-time definition of the `success` feature needs verification.
- A false-positive concern was observed with `example.com`.
- DNS and WHOIS lookups may return incomplete information.
- Model confidence is not a guarantee of correctness.

## Future Scope
- Align all training and inference feature definitions.
- Validate predictions on independently verified URLs.
- Reduce false positives.
- Improve security analysis and monitoring.
- Expand algorithm comparison and evaluation.

## Disclaimer
PhishGuard is an academic machine-learning project. Predictions are estimates and should not be treated as definitive proof that a website is safe or malicious.

## Developer
Megha Verma  
B.Tech – Artificial Intelligence and Machine Learning  
The NorthCap University