from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/prediction')
def prediction():
    return render_template('prediction.html') 



@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('prediction.html')
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Age=request.form.get('Age'),
            Location=request.form.get('Location'),
            Subscription_Length_Months=request.form.get('Subscription_Length_Months'),
            Monthly_Bill=float(request.form.get('Monthly_Bill')),
            Total_Usage_GB=float(request.form.get('Total_Usage_GB'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline() 
        print("Mid Prediction")
        print(pred_df.columns)
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('prediction.html',results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) 
