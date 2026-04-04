import shap
from visa_approval_prediction.logger import logging


class VisaShapExplainer:
    """Generates SHAP-based explanations for visa approval predictions."""

    def __init__(self, model):
        """
        Args:
            model: visaModel with preprocessing_object (ColumnTransformer) and
                   trained_model_object (tree-based classifier).
        """
        self.preprocessor = model.preprocessing_object
        self.tree_model = model.trained_model_object
        self.explainer = shap.TreeExplainer(self.tree_model)
        self.feature_mapping = self._build_feature_mapping()
        logging.info(
            f"SHAP explainer initialized "
            f"(model={type(self.tree_model).__name__}, "
            f"transformed_features={len(self.feature_mapping)})"
        )

    def _build_feature_mapping(self):
        """Map each transformed feature index back to its original feature name."""
        mapping = []

        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder':
                continue

            if name == 'OneHotEncoder':
                for i, col in enumerate(columns):
                    n_cats = len(transformer.categories_[i])
                    mapping.extend([col] * n_cats)
            elif name == 'Transformer':
                for col in columns:
                    mapping.append(col)
            else:
                for col in columns:
                    mapping.append(col)

        return mapping

    def explain(self, input_df, input_data, prediction_result):
        """Generate SHAP-based explanation in the same format as analyze_features()."""
        transformed = self.preprocessor.transform(input_df)

        raw_shap = self.explainer.shap_values(transformed)

        # XGBoost returns a single ndarray; RandomForest returns [class0, class1]
        if isinstance(raw_shap, list):
            shap_values = raw_shap[1][0]
        else:
            shap_values = raw_shap[0]

        # Aggregate transformed-feature SHAP values back to original features.
        feature_shap = {}
        for idx, feature_name in enumerate(self.feature_mapping):
            if idx < len(shap_values):
                feature_shap[feature_name] = (
                    feature_shap.get(feature_name, 0.0) + shap_values[idx]
                )

        # Sort by absolute SHAP magnitude (most impactful first)
        sorted_features = sorted(
            feature_shap.items(), key=lambda x: abs(x[1]), reverse=True
        )

        strengths = []
        weaknesses = []
        suggestions = []

        # Target encoding: Certified=0, Denied=1
        # Positive SHAP -> pushes toward Denied (weakness)
        # Negative SHAP -> pushes toward Certified (strength)
        for feature_name, shap_val in sorted_features:
            if abs(shap_val) < 0.01:
                continue

            msg = self._format_message(feature_name, shap_val, input_data)

            if shap_val < 0:
                strengths.append(msg)
            else:
                weaknesses.append(msg)
                suggestion = self._get_suggestion(feature_name, input_data)
                if suggestion:
                    suggestions.append(suggestion)

        n_strong = sum(1 for _, v in sorted_features if v < -0.3)
        confidence_label = (
            "high" if n_strong >= 4 else "moderate" if n_strong >= 2 else "low"
        )

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'suggestions': suggestions,
            'confidence_label': confidence_label,
        }

    def _get_suggestion(self, feature_name, input_data):
        """Return an input-aware suggestion for a weakness."""
        value = input_data.get(feature_name, '')

        if feature_name == 'education_of_employee':
            if value in ("Master's", "Doctorate"):
                return None  # already high education, no suggestion
            elif value == "Bachelor's":
                return "A Master's or Doctorate degree would strengthen the application"
            else:
                return "A Bachelor's or higher degree improves approval chances"
        elif feature_name == 'has_job_experience':
            return "Gaining relevant work experience strengthens applications"
        elif feature_name == 'requires_job_training':
            if value == "Y":
                return "Applicants not requiring training have higher approval rates"
            return None
        elif feature_name == 'prevailing_wage':
            return "Higher-paying positions correlate with better approval odds"
        elif feature_name == 'no_of_employees':
            return "Larger employers tend to have smoother PERM processes"
        elif feature_name == 'full_time_position':
            if value != "Y":
                return "Full-time positions demonstrate stronger employer commitment"
            return None
        elif feature_name == 'company_age':
            return "More established companies have stronger approval track records"
        return None

    def _format_message(self, feature_name, shap_val, input_data):
        """Create a human-readable message for a single feature's SHAP contribution."""
        intensity = self._get_intensity(abs(shap_val))
        value = input_data.get(feature_name, '')
        # Negative SHAP -> Certified (favors), Positive -> Denied (works against)
        direction = 'favors' if shap_val < 0 else 'works against'

        if feature_name == 'education_of_employee':
            return f"{value} education {intensity} {direction} approval"
        elif feature_name == 'has_job_experience':
            exp = "Having" if value == "Y" else "Not having"
            return f"{exp} job experience {intensity} {direction} approval"
        elif feature_name == 'requires_job_training':
            trn = "Requiring" if value == "Y" else "Not requiring"
            return f"{trn} job training {intensity} {direction} approval"
        elif feature_name == 'full_time_position':
            pos = "Full-time" if value == "Y" else "Part-time"
            return f"{pos} position {intensity} {direction} approval"
        elif feature_name == 'no_of_employees':
            try:
                return f"Company size ({int(value):,} employees) {intensity} {direction} approval"
            except (ValueError, TypeError):
                return f"Company size {intensity} {direction} approval"
        elif feature_name == 'company_age':
            return f"Company age ({value} years) {intensity} {direction} approval"
        elif feature_name == 'prevailing_wage':
            try:
                return f"Prevailing wage (${float(value):,.0f}) {intensity} {direction} approval"
            except (ValueError, TypeError):
                return f"Prevailing wage {intensity} {direction} approval"
        elif feature_name == 'continent':
            return f"Applicant from {value} {intensity} {direction} approval"
        elif feature_name == 'region_of_employment':
            return f"Employment in {value} {intensity} {direction} approval"
        elif feature_name == 'unit_of_wage':
            return f"Wage unit ({value}) {intensity} {direction} approval"
        else:
            return f"{feature_name} ({value}) {intensity} {direction} approval"

    @staticmethod
    def _get_intensity(abs_shap):
        if abs_shap > 1.0:
            return 'strongly'
        elif abs_shap > 0.3:
            return 'moderately'
        return 'slightly'
