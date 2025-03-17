import sys 
import os 
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.myproject.exception import CustomException
from src.myproject.logger import logging
from src.myproject.components.data_ingestion import DataIngestion
from src.myproject.utils import read_sql_data
from src.myproject.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        This function is responsible for Data Transformation
        
        """
        try:
           
            logging.info("Data Transformation has been initiated")
            logging.info("Reading from MySQL Database")
            numeric_features = ['age','bp','bgr','bu','hemo']
            categorical_features = ['htn']
            num_pipeline = Pipeline(steps = [("imputer", SimpleImputer(strategy = 'median')),
                                             ('scalar', StandardScaler())
                                             ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy = 'most_frequent')),
                ('OneHot Encoder', OneHotEncoder(handle_unknown = 'ignore'))
            ])

            logging.info(f"Numeric Features: {numeric_features}")
            logging.info(f"Categorical Features: {categorical_features}")

            preprocessor = ColumnTransformer([
                ('Numerical Pipeline', num_pipeline, numeric_features),
                ('Categorical Pipeline', cat_pipeline, categorical_features)
            ])
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for initiating the data transformation
        
        """
        try:
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Data Transformation has been initiated")
            logging.info("Transforming the data")

            logging.info(f"Train Data Columns: {train_data.columns}")
            logging.info(f"Test Data Columns: {test_data.columns}")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'classification' 
            if target_column_name not in train_data.columns or target_column_name not in test_data.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in the data")
            #numeric_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'O']

            input_features_train_df = train_data.drop(columns=["id", target_column_name])
            target_feature_train_df = train_data[target_column_name] 

            input_feature_test_df = test_data.drop(columns=["id", target_column_name])
            target_feature_test_df = test_data[target_column_name]
            logging.info("Applying Preprocessing on Train Data and Test Data")
            
             # Encode the target variable
            label_encoder = LabelEncoder()
            target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = label_encoder.transform(target_feature_test_df)

            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]



            
            logging.info(f"Saved Preprocessor Object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr ,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
