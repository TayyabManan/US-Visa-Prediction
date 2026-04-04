# Project Report: US Visa Approval Prediction System

## CRISP-DM Framework Mapping

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. The table below maps each CRISP-DM phase to the corresponding report sections.

| CRISP-DM Phase | Report Section | Key Activities |
|----------------|---------------|----------------|
| **Business Understanding** | Section 1 | Defined the problem (PERM prediction + explainability), identified stakeholders (applicants, attorneys), established success criteria (F1 score, actionable SHAP insights) |
| **Data Understanding** | Sections 2 & 3 | Profiled the EasyVisa dataset (25,480 records), analyzed class imbalance (66.8/33.2), examined feature distributions and relationships |
| **Data Preparation** | Section 4 | Engineered `company_age`, selected encoding strategies per feature type, built scikit-learn ColumnTransformer pipeline, applied SMOTEENN to training set |
| **Modeling** | Section 5 | Evaluated Random Forest, Gradient Boosting, and XGBoost via GridSearchCV; built VotingEnsemble; tuned on F1 |
| **Evaluation** | Sections 5.4 & 5.5 | Confusion matrix analysis, per-class precision/recall/F1, confidence calibration assessment |
| **Deployment** | Section 7 | FastAPI backend, Docker containerization, Hugging Face Spaces hosting, SHAP-based explanation UI |

---

## 1. Problem Statement

When a U.S. employer wants to hire a foreign worker permanently, they must file a PERM (Program Electronic Review Management) labor certification with the Department of Labor. The DOL either **certifies** or **denies** each case. The process is opaque — applicants and immigration attorneys have limited visibility into which factors drive outcomes.

**Goal**: Build a machine learning system that predicts whether a PERM application will be certified or denied, and explains *why* — giving applicants actionable insight into their case strength before they file.

### Why This Matters

- PERM processing takes 6–18 months. A denial means restarting from scratch.
- Legal fees for a PERM filing range from $5,000–$15,000. Predicting denial risk upfront saves time and money.
- Explainability (not just a yes/no prediction) helps attorneys identify weak points and strengthen applications before submission.

---

## 2. Dataset

**Source**: EasyVisa dataset — historical PERM labor certification records (Phani Srinivas et al., published on Kaggle).

| Property | Value |
|----------|-------|
| Total records | 25,480 |
| Features | 12 columns (10 used for prediction) |
| Target | `case_status` (Certified / Denied) |
| Class distribution | 66.8% Certified (17,021) / 33.2% Denied (8,459) |

### 2.1 Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `case_id` | ID | Unique identifier (dropped before training) |
| `continent` | Categorical | Applicant's continent of origin (Asia, Europe, Africa, N. America, S. America, Oceania) |
| `education_of_employee` | Ordinal | Highest education level (High School, Bachelor's, Master's, Doctorate) |
| `has_job_experience` | Binary | Whether applicant has prior relevant work experience (Y/N) |
| `requires_job_training` | Binary | Whether the position requires additional training (Y/N) |
| `no_of_employees` | Numeric | Number of employees at the sponsoring company |
| `yr_of_estab` | Numeric | Year the company was established (transformed to `company_age`) |
| `region_of_employment` | Categorical | U.S. region where the job is located (West, Northeast, South, Midwest, Island) |
| `prevailing_wage` | Numeric | DOL-determined wage for the position |
| `unit_of_wage` | Categorical | Pay period (Hour, Week, Month, Year) |
| `full_time_position` | Binary | Whether the position is full-time (Y/N) |
| `case_status` | Target | Certified (0) or Denied (1) |

### 2.2 Dropped Columns

- **`case_id`**: Unique identifier with no predictive value.
- **`yr_of_estab`**: Replaced with `company_age` (current year minus establishment year) — a relative measure that doesn't become stale over time.

---

## 3. Exploratory Data Analysis

Full EDA code with interactive visualizations is available in `notebook/eda.ipynb`. Key figures referenced below:

- **Figure 1** (Section 3.1): Class distribution bar chart and pie chart — visualizes the 66.8/33.2 Certified/Denied split.
- **Figure 2** (Section 3.2): Certification rate by category for each categorical feature — side-by-side count plots and horizontal bar charts showing how approval rate varies across education levels, continents, regions, etc.
- **Figure 3** (Section 3.3): Numeric feature distributions — histograms, box plots by status, and KDE overlays for `no_of_employees`, `prevailing_wage`, and `company_age`.
- **Figure 4** (Section 3.3): Correlation heatmap — shows feature-to-feature and feature-to-target correlations.
- **Figure 5** (Section 5.4): Confusion matrix — heatmap visualization of the 2x2 prediction outcome matrix (see table below).
- **Figure 6** (Section 5.5): Model comparison bar chart — Denied F1 across Random Forest, Gradient Boosting, XGBoost, and VotingEnsemble.

### 3.1 Class Balance

The dataset has a **2.01:1 imbalance** — 17,021 Certified (66.8%) vs 8,459 Denied (33.2%). This is significant because:
- A naive classifier predicting "Certified" for every case achieves 66.8% accuracy while catching zero denials.
- We need a metric that penalizes missing denials. This ruled out accuracy as the primary metric (see Section 5.2).

### 3.2 Feature Analysis

**Education**: Strong ordinal relationship with approval. Doctorate holders have the highest certification rate; High School the lowest. This makes intuitive sense — PERM is designed for positions requiring specialized skills.

**Job Experience**: Applicants with prior experience (`Y`) are certified at a meaningfully higher rate. Experience signals that the worker already possesses the skills the employer claims to need.

**Job Training**: Requiring training (`Y`) correlates with higher denial rates. If the applicant needs training, DOL may question whether they truly meet the job requirements.

**Prevailing Wage**: Higher wages correlate with certification. High-wage positions are typically specialized roles where it's harder to find qualified U.S. workers — exactly what PERM is designed for.

**Company Size**: Larger employers have slightly higher certification rates, likely due to more established HR and legal processes for PERM filings.

**Full-Time Position**: Full-time roles are certified at a higher rate than part-time, suggesting DOL views full-time offers as more credible employer commitments.

**Continent**: Asia represents the largest applicant group. Approval rates vary by continent, though this likely reflects confounding factors (education levels, industry mix) rather than geographic bias per se. See Section 9 for ethical considerations.

**Region of Employment**: Regional variation exists but is modest. West and Northeast have the most filings, reflecting tech and finance industry concentration.

### 3.3 Numeric Feature Distributions

- **`no_of_employees`**: Heavily right-skewed. Most employers are small-to-mid size, with a long tail of large corporations. This skew motivated the power transformation in preprocessing.
- **`prevailing_wage`**: Varies dramatically by `unit_of_wage`. A $35/hour wage and a $72,800/year wage are roughly equivalent, but the raw numbers differ by 2,000x. The model handles this through the combination of `prevailing_wage` and `unit_of_wage` features.
- **`company_age`**: Right-skewed. Applied power transformation to normalize.

### 3.4 Missing Values

No missing values in the dataset. All 25,480 records are complete across all 12 columns.

---

## 4. Data Preprocessing

### 4.1 Feature Engineering

**`company_age`**: Computed as `current_year - yr_of_estab`. This relative feature ages gracefully — a company established in 1990 is "34 years old" in 2024, not "1990" (an arbitrary number to the model).

### 4.2 Encoding Strategy

Different encoding strategies were chosen based on feature semantics:

| Strategy | Features | Why |
|----------|----------|-----|
| **Ordinal Encoding** | `education_of_employee`, `has_job_experience`, `requires_job_training`, `full_time_position` | These have a natural order or binary values. Ordinal encoding preserves the ranking (High School < Bachelor's < Master's < Doctorate). |
| **One-Hot Encoding** | `continent`, `unit_of_wage`, `region_of_employment` | These are nominal — no inherent order. One-hot encoding avoids imposing a false ordinal relationship. |
| **Power Transform** | `no_of_employees`, `company_age` | Both are heavily right-skewed. Power transformation (Yeo-Johnson) normalizes the distribution, helping tree-based models make better split decisions. |
| **Passthrough** | `prevailing_wage` | Already well-distributed enough for tree models. No transformation needed. |

### 4.3 Preprocessing Pipeline

The preprocessing is implemented as a scikit-learn `ColumnTransformer` with three named transformers:
1. **`Transformer`**: PowerTransformer on `no_of_employees` and `company_age`
2. **`OrdinalEncoder`**: On `has_job_experience`, `requires_job_training`, `full_time_position`, `education_of_employee`
3. **`OneHotEncoder`**: On `continent`, `unit_of_wage`, `region_of_employment`

This pipeline is serialized alongside the model in `model.pkl` (wrapped in a `visaModel` object), ensuring identical preprocessing at inference time.

---

## 5. Model Training and Evaluation

### 5.1 Handling Class Imbalance: SMOTEENN

The 2:1 class imbalance (Certified:Denied) means the model sees twice as many positive cases during training. Without intervention, it learns to favor "Certified" predictions.

**SMOTEENN** was chosen over simpler alternatives:

| Method | What It Does | Why Not Sufficient |
|--------|-------------|-------------------|
| Random oversampling | Duplicates minority samples | Creates exact copies, leads to overfitting |
| SMOTE alone | Synthesizes new minority samples via interpolation | Can create noisy samples near the decision boundary |
| Random undersampling | Removes majority samples | Throws away useful data |
| **SMOTEENN** | SMOTE + Edited Nearest Neighbors cleanup | Synthesizes new minority samples, then removes noisy samples from both classes that are misclassified by their neighbors |

SMOTEENN was applied **only to the training set**. The test set was left untouched to reflect real-world class distribution. This prevents data leakage — evaluating on resampled test data would give inflated metrics.

### 5.2 Metric Selection: F1 Score

**Why not accuracy?**
A model predicting "Certified" for every case scores 66.8% accuracy. That's useless — it catches zero denials.

**Why F1 over precision or recall alone?**
- Pure precision: "Only predict denied when very sure" — misses many actual denials.
- Pure recall: "Flag everything as denied" — too many false alarms.
- **F1 balances both**: It's the harmonic mean of precision and recall, penalizing models that sacrifice one for the other.

F1 was used as the scoring metric in GridSearchCV, meaning hyperparameters were tuned to maximize this balance.

### 5.3 Model Selection

Three models were evaluated via GridSearchCV with 5-fold cross-validation:

| Model | Why Considered |
|-------|---------------|
| **Random Forest** | Robust baseline, handles mixed feature types well, resistant to overfitting |
| **Gradient Boosting** | Sequential error correction, often outperforms RF on structured data |
| **XGBoost** | Optimized gradient boosting with regularization, typically best-in-class for tabular data |

**Hyperparameter search space** (from `config/model.yaml`):

- **Random Forest**: `n_estimators` [100, 200, 300], `max_depth` [10, 20, 30], `max_features` [sqrt, log2]
- **Gradient Boosting**: `n_estimators` [100, 200, 300], `learning_rate` [0.05, 0.1, 0.2], `max_depth` [3, 5, 7]
- **XGBoost**: `n_estimators` [100, 200, 300], `learning_rate` [0.05, 0.1, 0.2], `max_depth` [3, 5, 7]

After individual evaluation, the best-performing models were combined into a **soft-voting ensemble** (`VotingClassifier` with `voting='soft'`), which averages class probability outputs from each model. The ensemble was selected as the final model because it smooths out individual model weaknesses and produces better-calibrated probability estimates.

### 5.4 Results

The final **VotingEnsemble** achieved a **weighted F1 of 76%** on the held-out test set (20% stratified split, 5,096 samples).

**Confusion Matrix**:

|  | Predicted Certified | Predicted Denied | Total |
|---|---:|---:|---:|
| **Actual Certified** | 3,026 (TN) | 378 (FP) | 3,404 |
| **Actual Denied** | 422 (FN) | 1,270 (TP) | 1,692 |
| **Total** | 3,448 | 1,648 | 5,096 |

**Per-Class Metrics**:

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Certified | 87.8% | 88.9% | 88.3% | 3,404 |
| Denied | 77.1% | 75.1% | 76.0% | 1,692 |
| **Weighted Avg** | **84.2%** | **84.3%** | **84.2%** | **5,096** |

**Reading the confusion matrix**:
- **3,026 True Negatives**: Certified cases correctly predicted as Certified.
- **1,270 True Positives**: Denied cases correctly predicted as Denied — the model catches 75.1% of actual denials.
- **378 False Positives**: Certified cases wrongly flagged as Denied — 11.1% of certified applications receive a false alarm. For an advisory tool, this is acceptable: a false "denied" prediction with low confidence and weak SHAP reasons is easily dismissed.
- **422 False Negatives**: Denied cases the model missed — predicted Certified but were actually Denied. This is the costliest error type. The 24.9% miss rate means roughly 1 in 4 denials goes undetected.

**Overall accuracy**: 84.3%, well above the 66.8% naive baseline.

### 5.5 Single Model vs. Ensemble Comparison

| Model | Denied F1 | Certified F1 | Weighted F1 |
|-------|-----------|-------------|-------------|
| Random Forest | 72.4% | 86.1% | 81.6% |
| Gradient Boosting | 74.8% | 87.5% | 83.3% |
| XGBoost | 75.3% | 87.9% | 83.7% |
| **VotingEnsemble** | **76.0%** | **88.3%** | **84.2%** |

XGBoost was the strongest individual model. The ensemble provided a modest but consistent improvement (+0.7% Denied F1 over XGBoost alone). The ensemble was selected as the final model for two reasons:
1. Soft voting averages probabilities across models, producing smoother and better-calibrated confidence scores.
2. The marginal F1 gain, while small, comes at negligible inference cost since all three models are fast on CPU.

---

## 6. Explainability

A prediction alone ("approved" or "denied") isn't useful to an applicant. They need to know **why** — which factors helped, which hurt, and what they can change.

### 6.1 SHAP TreeExplainer

**SHAP (SHapley Additive exPlanations)** was chosen for per-prediction explanations. Specifically, `TreeExplainer` — a fast, exact algorithm for tree-based models.

**How it works**: For each prediction, SHAP computes a value for every feature, representing how much that feature pushed the prediction toward "Certified" or "Denied" relative to the average case.

**Implementation challenge**: The preprocessing pipeline transforms 10 raw features into ~20 encoded features (due to one-hot encoding). SHAP operates on these transformed features. We built a mapping (`_build_feature_mapping` in `shap_explainer.py`) that aggregates transformed-feature SHAP values back to their original feature names. For example, the 6 one-hot columns for `continent` are summed back into a single SHAP value for "continent."

**Interpretation**:
- Negative SHAP value → pushes toward Certified (strength)
- Positive SHAP value → pushes toward Denied (weakness)
- Magnitude indicates strength: >1.0 = strong, 0.3–1.0 = moderate, <0.3 = slight

### 6.2 Rule-Based Fallback

SHAP can occasionally fail (edge cases in model structure, unseen feature combinations). A rule-based fallback (`analyze_features()` in `prediction_pipeline.py`) provides heuristic explanations based on known patterns in the data:
- Education level thresholds
- Wage benchmarks (annualized)
- Company size and age breakpoints
- Binary feature impacts

This ensures every prediction receives an explanation, even if SHAP encounters an error.

### 6.3 Confidence Scoring

The model outputs class probabilities via `predict_proba()`. The confidence score is `max(P(Certified), P(Denied)) * 100`.

The UI communicates confidence levels:
- **>80%**: High confidence — prediction is reliable
- **60–80%**: Moderate confidence — interpret with caution
- **<60%**: Low confidence — borderline case, prediction may be unreliable

---

## 7. Deployment

### 7.1 Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI + Uvicorn | Async API with automatic request validation |
| Frontend | Jinja2 + vanilla JS | Single-page prediction UI with SHAP results |
| Model serving | Pickle + class-level cache | Load once, serve many requests |
| Container | Docker (python:3.12-slim) | Reproducible deployment environment |
| Hosting | Hugging Face Spaces | Free Docker SDK hosting, port 7860 |

### 7.2 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the prediction UI |
| `/predict` | POST | JSON API — accepts 10 features, returns prediction + confidence + SHAP insights |
| `/` | POST | Form submission fallback — returns rendered HTML with result |

### 7.3 Model Serving

The model is loaded once via `visaModel` (defined in `entity/estimator.py`) and cached as a class variable. The `visaModel` object wraps both the preprocessing `ColumnTransformer` and the trained `VotingClassifier`, ensuring end-to-end consistency from raw input to prediction. The SHAP explainer is similarly cached after first use.

---

## 8. Design Decisions and Trade-offs

### 8.1 Why Tree-Based Models Over Deep Learning?

- **Dataset size**: 25,480 records is small for deep learning. Tree-based models consistently outperform neural networks on tabular data at this scale (Grinsztajn et al., 2022).
- **Interpretability**: SHAP TreeExplainer provides exact, fast explanations for tree models. Neural network explanations (LIME, gradient-based) are approximations and significantly slower.
- **Training time**: The full GridSearchCV pipeline trains in under 10 minutes on CPU. No GPU infrastructure needed.

### 8.2 Why F1 Over F2?

F2 weights recall higher than precision — useful when missing a denial is much worse than a false alarm. We considered both:
- **F1 = 76%**: Balanced precision (77.1%) and recall (75.1%) for the Denied class.
- **F2**: Would push recall higher at the cost of more false alarms on Certified cases.

For an informational tool (not a decision system), F1's balance is appropriate. Users see a confidence score and detailed SHAP analysis — a false alarm with low confidence and weak reasons is easily dismissed by the user.

### 8.3 Why SMOTEENN Over Class Weights?

Both address class imbalance. SMOTEENN creates synthetic minority samples and cleans noisy boundary samples. Class weighting (e.g., `scale_pos_weight` in XGBoost) adjusts the loss function.

SMOTEENN was chosen because:
- It physically balances the dataset, giving the model equal exposure to both classes
- The ENN cleanup step removes noisy samples near the decision boundary, improving generalization
- Empirically, it produced better F1 than `scale_pos_weight` alone on this dataset

### 8.4 Why FastAPI Over Flask?

Both are Python web frameworks capable of serving the prediction API. FastAPI was chosen for several reasons:

- **Async-native**: FastAPI is built on Starlette and supports `async/await` natively. Model loading and SHAP computation can be CPU-intensive; async request handling prevents one slow prediction from blocking the entire server.
- **Automatic validation**: Pydantic models (`PredictRequest`) validate incoming JSON against the expected schema and return clear 422 errors for malformed requests — no manual validation code needed.
- **Built-in OpenAPI docs**: FastAPI auto-generates interactive API documentation at `/docs`, useful for testing the `/predict` endpoint during development without a frontend.
- **Performance**: FastAPI consistently benchmarks faster than Flask for JSON APIs due to its ASGI foundation vs Flask's WSGI.

Flask would have worked, but FastAPI provided validation, documentation, and async support with less boilerplate.

---

## 9. Ethical Considerations

### 9.1 Continent as a Feature

The model uses `continent` (applicant's continent of origin) as an input feature. This is effectively a proxy for nationality and ethnicity, raising fairness concerns in a visa prediction context.

**Why it was retained**: Continent correlates with approval rates in the historical data. Removing it would reduce model accuracy without eliminating the bias — other features (education system, wage levels) partially encode the same information. Excluding the feature would make the model less accurate without making it more fair.

**Mitigation through transparency**: SHAP explanations explicitly surface when continent is influencing a prediction. If a user sees "Applicant from [continent] moderately works against approval," they can recognize this as a data pattern rather than a recommendation. This transparency allows users to critically evaluate the model's reasoning rather than accepting it blindly.

**Limitations**: The model reflects historical DOL decision patterns, which may contain systemic biases. It should never be used as a decision-making tool — only as an informational aid. We recommend that any production deployment include regular fairness audits using SHAP to monitor whether continent-based disparities are growing or shrinking over time.

### 9.2 Broader Concerns

- **Not legal advice**: The UI explicitly states this is an informational tool, not a guarantee of outcome.
- **Feedback loops**: If attorneys use the tool to selectively file "likely certified" cases, the resulting data could reinforce existing biases.
- **Access equity**: The tool is freely hosted to avoid creating an information advantage only for well-resourced applicants.

---

## 10. Limitations

1. **Historical data only**: The model reflects patterns from past DOL decisions. Policy changes, new regulations, or shifting priorities are not captured.

2. **Limited feature set**: Only 10 features are used. Real PERM applications involve hundreds of fields (job description, recruitment steps, attorney information, etc.). The model captures broad patterns, not case-specific nuances.

3. **No temporal awareness**: The model treats all historical records equally. Recent trends may differ from older patterns.

4. **Confidence calibration**: The `predict_proba` values from the VotingClassifier are not perfectly calibrated probabilities. The confidence percentage is indicative, not a true probability.

5. **Not legal advice**: PERM outcomes depend on factors beyond what any model can capture (quality of documentation, specific DOL officer review, recruitment evidence, etc.).

---

## 11. Future Improvements

- **Probability calibration**: Apply Platt scaling or isotonic regression to produce better-calibrated confidence scores.
- **Feature expansion**: Incorporate job title/SOC code, NAICS industry code, and wage ratio (offered wage / prevailing wage) if data becomes available.
- **Temporal weighting**: Weight recent cases higher during training to capture evolving DOL decision patterns.
- **Fairness auditing**: Implement automated SHAP-based fairness metrics to monitor continent-driven disparities across model versions.
- **A/B testing**: Compare SHAP explanations with rule-based explanations in user studies to measure which format is more actionable.
- **Retraining pipeline**: Automated retraining when new DOL disclosure data is published (quarterly).

---

## 12. Team Contributions

**Group 3** — US Visa Approval Prediction System using Machine Learning, MLOps & AWS

The project was divided into 7 modules. Each team member owned a primary module and contributed to additional modules as listed below.

| Member | Primary Module | All Modules | Key Contributions |
|--------|---------------|-------------|-------------------|
| Muhammad Asim | Module 1: Data Collection & Preprocessing | 1, 5 & 6 | Data pipeline (ingestion, validation, transformation), feature engineering (`company_age`), encoding strategy, SMOTEENN resampling, cloud infrastructure (AWS S3, Docker), MLOps & CI/CD pipeline |
| Syed Measum | Module 2: Machine Learning | 1, 2 & 7 | Model training (Random Forest, Gradient Boosting, XGBoost), GridSearchCV hyperparameter tuning, VotingEnsemble construction, evaluation metrics (confusion matrix, F1), data collection support, documentation |
| Muhammad Tayyab | Module 3: Backend API | 3, 4 & 7 | FastAPI application (`app.py`), `/predict` endpoint, Pydantic request validation, model serving & caching (`visaModel`), frontend interface support, documentation |
| Bilal | Module 4: Frontend Interface | 5, 6 & 7 | Prediction UI (`visa.html`), step-by-step loading animation, SHAP results display (strengths/weaknesses/suggestions), responsive design, cloud infrastructure support, CI/CD support, documentation |

**Module breakdown**:

| Module | Description |
|--------|-------------|
| Module 1 | Data Collection & Preprocessing — dataset ingestion, schema validation, feature engineering, encoding pipelines |
| Module 2 | Machine Learning — model selection, training, hyperparameter tuning, SMOTEENN, evaluation |
| Module 3 | Backend API — FastAPI endpoints, request validation, model serving, SHAP integration |
| Module 4 | Frontend Interface — prediction UI, result visualization, loading states, responsive design |
| Module 5 | Cloud Infrastructure — AWS S3, Docker containerization, Hugging Face Spaces deployment |
| Module 6 | MLOps & CI/CD — automated pipeline, model versioning, deployment automation |
| Module 7 | Documentation — project report, EDA notebook, README, code documentation |

---

## 13. References

1. **EasyVisa Dataset** — Phani Srinivas et al. Historical PERM labor certification records. Available on Kaggle. https://www.kaggle.com/datasets/moro23/easyvisa-dataset

2. **SHAP (SHapley Additive exPlanations)** — Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*. https://github.com/shap/shap

3. **SMOTEENN** — Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data. *ACM SIGKDD Explorations Newsletter*, 6(1), 20–29. Implementation: imbalanced-learn. https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html

4. **XGBoost** — Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. https://xgboost.readthedocs.io/

5. **scikit-learn** — Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://scikit-learn.org/

6. **FastAPI** — Ramirez, S. FastAPI framework, high performance, easy to learn, fast to code. https://fastapi.tiangolo.com/

7. **Hugging Face Spaces** — Hugging Face. Docker SDK for Spaces deployment. https://huggingface.co/docs/hub/spaces-sdks-docker

8. **CRISP-DM** — Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a Standard Process Model for Data Mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*.

9. **Tree-based models vs. deep learning on tabular data** — Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems (NeurIPS)*. https://arxiv.org/abs/2207.08815
