import sys 
import pandas as pd
import numpy as np

from src.myproject.exception import CustomException 
from src.myproject.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 age: int,
                 bp: int,
                 bgr: int,
                 bu:str,
                 hemo: int,
                 htn: str):
        self.age = age
        self.bp = bp
        self.bgr = bgr
        self.bu = bu
        self.hemo = hemo
        self.htn = htn


    def get_data_as_dataframe(self):
        try:
            custom_data = {
                'age': [self.age],
                'bp': [self.bp],
                'bgr': [self.bgr],
                'bu': [self.bu],
                'hemo': [self.hemo],
                'htn': [self.htn]
            }
            return pd.DataFrame(custom_data)
        except Exception as e:
            raise CustomException(e,sys)
        