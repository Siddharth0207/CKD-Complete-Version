import sys 
import os 
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix

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
            numeric_features = ['age','bp','bgr','hemo']
            categorical_features = ['htn','bu']
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

            input_features_train_df = train_data.drop(columns=["id", target_column_name])
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(columns=["id", target_column_name])
            target_feature_test_df = test_data[target_column_name]

            logging.info("Applying Preprocessing on Train Data and Test Data")

            #print("target_feature_train_df shape:", target_feature_train_df.shape)
            #print("target_feature_train_df head:", target_feature_train_df.head())

            label_encoder = LabelEncoder()
            target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = label_encoder.transform(target_feature_test_df)

            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df) # Move this line up.
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

             # Convert sparse matrix to dense array
            if isinstance(input_feature_train_arr, csr_matrix):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if isinstance(input_feature_test_arr, csr_matrix):
                input_feature_test_arr = input_feature_test_arr.toarray()
            
            #print("input_feature_train_arr shape:", input_feature_train_arr.shape) # Move this line down.
            #print("input_feature_train_arr head:", input_feature_train_arr[:5])

            target_feature_train_encoded = target_feature_train_encoded.reshape(-1, 1)
            target_feature_test_encoded = target_feature_test_encoded.reshape(-1, 1)

            #print("target_feature_train_encoded shape:", target_feature_train_encoded.shape)

            train_arr = np.hstack((input_feature_train_arr, target_feature_train_encoded))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_encoded))

            logging.info(f"Saved Preprocessor Object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            
