import os 
import sys 
from src.myproject.exception import CustomException
from src.myproject.logger import logging

import pandas as pd 
from dotenv import load_dotenv
import pymysql
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


import pickle
import numpy as np 
import pandas as pd


#Loading the Data from .env file inorder to use the same credentials  in all the files
logging.info("Loading the Data from .env file")
load_dotenv()
host = os.getenv('host')
user = os.getenv('user')
password= os.getenv('password')
db = os.getenv('db')




def read_sql_data():
    logging.info("Reading the Data from MySQL has started.")
    # this will basically retutn the dataset from the database
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection has been established")
        df =  pd.read_sql_query('Select * from kidney_disease', mydb)
        print(df.head())

        return df


    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    """
    This function is responsible for saving the object as a pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object has been saved at {file_path}")
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)