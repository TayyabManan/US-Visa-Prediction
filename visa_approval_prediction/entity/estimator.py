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
    
    