import os
import sys

import pandas as pd
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging
from pandas import DataFrame

class visaData:
    def __init__(self,
                continent,
                education_of_employee,
                has_job_experience,
                requires_job_training,
                no_of_employees,
                region_of_employment,
                prevailing_wage,
                unit_of_wage,
                full_time_position,
                company_age
                ):
        """
        visa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age


        except Exception as e:
            raise visaException(e, sys) from e

    def get_visa_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from visaData class input
        """
        try:
            
            visa_input_dict = self.get_visa_data_as_dict()
            return DataFrame(visa_input_dict)
        
        except Exception as e:
            raise visaException(e, sys) from e


    def get_visa_data_as_dict(self):
        """
        This function returns a dictionary from visaData class input 
        """
        logging.info("Entered get_visa_data_as_dict method as visaData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created visa data dict")

            logging.info("Exited get_visa_data_as_dict method as visaData class")

            return input_data

        except Exception as e:
            raise visaException(e, sys) from e

def analyze_features(data: dict, result: str) -> dict:
    """Analyze input features and return strengths, weaknesses, and suggestions."""
    strengths = []
    weaknesses = []
    suggestions = []

    # Education
    edu = data.get("education_of_employee", "")
    if edu in ("Master's", "Doctorate"):
        strengths.append(f"{edu} degree strongly favors approval")
    elif edu == "Bachelor's":
        strengths.append("Bachelor's degree meets typical requirements")
    else:
        weaknesses.append("High School education significantly lowers approval odds")
        suggestions.append("A Bachelor's or higher degree greatly improves chances")

    # Job experience
    if data.get("has_job_experience") == "Y":
        strengths.append("Prior job experience is a strong positive signal")
    else:
        weaknesses.append("No prior job experience weakens the application")
        suggestions.append("Gaining relevant work experience before applying improves outcomes")

    # Job training
    if data.get("requires_job_training") == "Y":
        weaknesses.append("Requiring job training suggests the applicant isn't fully qualified")
        suggestions.append("Applicants who don't require training have higher approval rates")
    else:
        strengths.append("No job training required indicates readiness for the role")

    # Full time
    if data.get("full_time_position") == "Y":
        strengths.append("Full-time positions have higher approval rates than part-time")
    else:
        weaknesses.append("Part-time positions are approved at a lower rate")
        suggestions.append("Full-time positions demonstrate stronger employer commitment")

    # Prevailing wage
    try:
        wage = float(data.get("prevailing_wage", 0))
        unit = data.get("unit_of_wage", "Year")
        annual = wage
        if unit == "Hour":
            annual = wage * 2080
        elif unit == "Week":
            annual = wage * 52
        elif unit == "Month":
            annual = wage * 12

        if annual >= 80000:
            strengths.append(f"Competitive wage (~${annual:,.0f}/yr) signals a skilled position")
        elif annual >= 45000:
            pass  # Neutral, don't mention
        else:
            weaknesses.append(f"Lower wage (~${annual:,.0f}/yr) is associated with higher denial rates")
            suggestions.append("Higher-paying positions correlate with better approval odds")
    except (ValueError, TypeError):
        pass

    # Company size
    try:
        employees = int(data.get("no_of_employees", 0))
        if employees >= 25000:
            strengths.append("Large employer with established hiring processes")
        elif employees >= 18000:
            pass  # Neutral
        else:
            weaknesses.append("Smaller employers face slightly more scrutiny in PERM cases")
    except (ValueError, TypeError):
        pass

    # Company age
    try:
        age = int(data.get("company_age", 0))
        if age >= 50:
            strengths.append(f"Well-established company ({age} years) adds credibility")
        elif age < 25:
            weaknesses.append("Relatively young company may face additional scrutiny")
            suggestions.append("Established companies (50+ years) have stronger approval track records")
    except (ValueError, TypeError):
        pass

    # Continent
    continent = data.get("continent", "")
    if continent == "Asia":
        pass  # Largest applicant pool, neutral
    elif continent in ("Europe", "North America", "Oceania"):
        strengths.append(f"Applicants from {continent} have competitive approval rates")

    confidence_label = "high" if len(strengths) >= 4 else "moderate" if len(strengths) >= 2 else "low"

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "confidence_label": confidence_label,
    }


LOCAL_MODEL_PATH = os.path.join("artifact", "model.pkl")

class visaClassifier:
    _cached_model = None
    _cached_explainer = None

    def __init__(self) -> None:
        pass

    @classmethod
    def _load_model(cls):
        if cls._cached_model is None:
            import pickle
            logging.info(f"Loading model from {LOCAL_MODEL_PATH}")
            with open(LOCAL_MODEL_PATH, "rb") as f:
                cls._cached_model = pickle.load(f)
            logging.info("Model loaded successfully")
        return cls._cached_model

    def predict(self, dataframe) -> str:
        try:
            model = self._load_model()
            return model.predict(dataframe)
        except Exception as e:
            raise visaException(e, sys)

    def predict_with_confidence(self, dataframe):
        try:
            model = self._load_model()
            prediction = model.predict(dataframe)[0]
            proba = model.predict_proba(dataframe)
            confidence = None
            if proba is not None:
                confidence = float(max(proba[0])) * 100
            return prediction, confidence
        except Exception as e:
            raise visaException(e, sys)

    @classmethod
    def _get_explainer(cls):
        if cls._cached_explainer is None:
            from visa_approval_prediction.explainability.shap_explainer import VisaShapExplainer
            model = cls._load_model()
            cls._cached_explainer = VisaShapExplainer(model)
        return cls._cached_explainer

    def explain(self, input_df, input_data, prediction_result):
        """Generate SHAP-based insights, falling back to rule-based if SHAP fails."""
        try:
            explainer = self._get_explainer()
            return explainer.explain(input_df, input_data, prediction_result)
        except Exception as e:
            logging.warning(f"SHAP explanation failed ({e}), falling back to rule-based")
            return analyze_features(input_data, prediction_result)