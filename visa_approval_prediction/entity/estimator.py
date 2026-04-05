import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.Certified:int = 0
        self.Denied:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    
    

class ThresholdClassifier:
    """Wraps a trained model and applies a custom probability threshold for predict()."""
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold

    def predict(self, X):
        import numpy as np
        proba = self.base_model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    @property
    def classes_(self):
        return self.base_model.classes_

    def __repr__(self):
        return f"ThresholdClassifier({type(self.base_model).__name__}, threshold={self.threshold:.3f})"

    def __str__(self):
        return self.__repr__()


class visaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        logging.info("Entered predict method of visaModel class")
        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise visaException(e, sys) from e

    def predict_proba(self, dataframe: DataFrame):
        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)
            if hasattr(self.trained_model_object, 'predict_proba'):
                return self.trained_model_object.predict_proba(transformed_feature)
            return None
        except Exception as e:
            raise visaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    