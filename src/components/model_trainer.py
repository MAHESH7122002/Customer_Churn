import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingRegressor,
    
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                # "Decision Tree": DecisionTreeClassifier(),
                # "Logistic Regression": LogisticRegression(),
                # "XGBClassifier": XGBClassifier(),
            }
            params={
                "Decision Tree": {
                    'max_depth': 12,
                    'max_features': "sqrt"
                },
                "Random Forest":{
                    'n_estimators':500,
                    'max_depth':15,
                    'min_samples_split':5
                },
                "XGBClassifier":{
                    'n_estimators': 500,
                    'max_depth': 16,
                },

                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            # print(best_model_score)
            # if best_model_score<0.4:
            #     raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            



            
        except Exception as e:
            raise CustomException(e,sys)
        