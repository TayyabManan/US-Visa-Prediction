---
title: US Visa Approval Predictor
emoji: "\U0001F6C2"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# US Visa Approval Predictor

Predicts the likelihood of PERM labor certification approval using employer, applicant, and position data from historical DOL records.

**Live demo**: [Hugging Face Spaces](https://huggingface.co/spaces/TayyabManan/visa_prediction)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

| | |
|---|---|
| **Dataset** | EasyVisa — 25,480 historical PERM records |
| **Features** | 10 input features (continent, education, wage, employer info, etc.) |
| **Model** | Gradient Boosting with stacking ensemble selection + threshold tuning |
| **Accuracy** | 73.2% on unseen test data (denied recall: 61.4%) |
| **Explainability** | SHAP TreeExplainer with rule-based fallback |
| **Class Split** | 66.8% Certified / 33.2% Denied |

## Features

- **Prediction with confidence score** — returns approved/denied with probability percentage
- **SHAP explanations** — per-prediction breakdown of which factors helped or hurt the case
- **Profile analysis** — strengths, weaknesses, and actionable suggestions
- **Step-by-step loading UI** — animated progress through validation, model loading, prediction, SHAP computation, and analysis

## Project Structure

```
.
├── app.py                          # FastAPI application
├── train_model.py                  # Modal GPU training script (H100)
├── Dockerfile                      # Docker build for deployment
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── artifact/
│   └── model.pkl                   # Trained model (Gradient Boosting + threshold tuning + preprocessing pipeline)
├── config/
│   ├── model.yaml                  # Model hyperparameters
│   └── schema.yaml                 # Data schema definition
├── notebook/
│   ├── 1_Exploratory_Data_Analysis.ipynb
│   ├── 2_Feature_Engineering_and_Model_Selection.ipynb
│   └── 3_Model_Evaluation.ipynb
├── figures/                        # Saved plots for the project report (fig1–fig9)
├── templates/
│   └── visa.html                   # Main UI (single-page app)
└── visa_approval_prediction/
    ├── constants/                  # App config (host, port, file paths)
    ├── entity/
    │   └── estimator.py            # visaModel + ThresholdClassifier
    ├── exception/                  # Custom exception handling
    ├── explainability/
    │   └── shap_explainer.py       # SHAP TreeExplainer integration
    ├── logger/                     # Logging setup
    └── pipeline/
        └── prediction_pipeline.py  # visaData, visaClassifier, rule-based fallback
```

## Input Features

| Feature | Type | Example Values |
|---------|------|----------------|
| Continent of Origin | Categorical | Asia, Europe, Africa, North America, South America, Oceania |
| Education Level | Categorical | High School, Bachelor's, Master's, Doctorate |
| Prior Job Experience | Binary | Yes / No |
| Requires Job Training | Binary | Yes / No |
| Employment Region | Categorical | West, Northeast, South, Midwest, Island |
| Employment Type | Binary | Full-time / Part-time |
| Prevailing Wage | Numeric | Dollar amount |
| Wage Unit | Categorical | Hour, Week, Month, Year |
| Number of Employees | Numeric | Company size |
| Company Age | Numeric | Years in operation |

## Run Locally

```bash
# Clone the repository
git clone https://github.com/TayyabManan/US-Visa-Prediction.git
cd US-Visa-Prediction

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt

# Start the server
python app.py
```

The app will be available at `http://localhost:7860`.

## Run with Docker

```bash
docker build -t visa-predictor .
docker run -p 7860:7860 visa-predictor
```

## API

### `POST /predict`

**Request body** (JSON):

```json
{
  "continent": "Asia",
  "education_of_employee": "Master's",
  "has_job_experience": "Y",
  "requires_job_training": "N",
  "no_of_employees": "5000",
  "company_age": "30",
  "region_of_employment": "West",
  "prevailing_wage": "95000",
  "unit_of_wage": "Year",
  "full_time_position": "Y"
}
```

**Response**:

```json
{
  "result": "approved",
  "confidence": 87.3,
  "insights": {
    "strengths": ["Master's education strongly favors approval", "..."],
    "weaknesses": [],
    "suggestions": [],
    "confidence_label": "high"
  }
}
```

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **ML**: Gradient Boosting, XGBoost, LightGBM, CatBoost, scikit-learn (stacking ensemble, threshold tuning)
- **Explainability**: SHAP TreeExplainer
- **Frontend**: Jinja2 templates (vanilla HTML/CSS/JS)
- **Deployment**: Docker, Hugging Face Spaces

## License

MIT
