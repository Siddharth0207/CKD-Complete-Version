from src.myproject.logger import logging
from src.myproject.exception import CustomException
import sys
from src.myproject.components.data_ingestion import DataIngestion
from src.myproject.components.data_ingestion import DataIngestionConfig
from src.myproject.components.data_transformation import DataTransformation
from src.myproject.components.data_transformation import DataTransformationConfig
from src.myproject.components.model_trainer import ModelTrainerConfig , ModelTrainer

if __name__ == '__main__':
    logging.info('The execution of the program has started')

    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr , test_arr,_= data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        ## Model Training
        model_trainer = ModelTrainer()
        r2_square , best_model_name , best_model_score, authenticity = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Best Model Name: {best_model_name}")
        print(f"Best Model Score: {best_model_score}")
        print(f"R2 Score: {r2_square}")
        print(f"Authenticity: {authenticity}")



        
    except Exception as e:
        logging.info("Custom Exception has been raised")
        raise CustomException(e,sys)