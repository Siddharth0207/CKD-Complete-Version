import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor 
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.myproject.exception import CustomException
from src.myproject.logger import logging
from src.myproject.utils import save_object , evaluate_models
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("SPlit training and test data input")
            label_encoder = LabelEncoder()
            
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier(),
            }
            param = {
                "Logistic Regression": {},
                "Random Forest": {},
                "Gradient Boosting": {},
                "AdaBoost": {},
                "KNN": {},
                "Decision Tree": {},
                "XGBoost": {},
            }
            
            model_report: dict = evaluate_models(
                X_train, y_train_encoded, X_test, y_test_encoded, models, param
            )
            logging.info("Model Training has been completed")

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model Found")
            logging.info(f"Best Model found on both training and testing dataset.")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            predicted_probabilities = best_model.predict_proba(X_test)[:, 1]
            predicted = (predicted_probabilities > 0.5).astype(int)

            authenticity = confusion_matrix(y_test_encoded, predicted)
            r2_square = r2_score(y_test_encoded, predicted)

            return r2_square, best_model_name, best_model_score, authenticity

        except Exception as e:
            raise CustomException(e, sys)
