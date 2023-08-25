import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading") 
            features['Gender'] = features['Gender'][0].title()
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            if preds==1.0:
                res = 'Churn'
            else:
                res = 'Not churn'
            return res
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Gender: str,
        Age: int,
        Location:str,
        Subscription_Length_Months: float,
        Monthly_Bill: float,
        Total_Usage_GB: float):

        self.Gender = Gender

        self.Age = Age

        self.Location = Location

        self.Subscription_Length_Months = Subscription_Length_Months

        self.Monthly_Bill = Monthly_Bill

        self.Total_Usage_GB = Total_Usage_GB

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Location":[self.Location],
                "Subscription_Length_Months": [self.Subscription_Length_Months],
                "Monthly_Bill": [self.Monthly_Bill],
                "Total_Usage_GB": [self.Total_Usage_GB],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
