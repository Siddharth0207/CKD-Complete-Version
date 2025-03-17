from flask import Flask, request, render_template
from src.myproject.logger import logging

# Set up logging


import numpy as np 
import pandas as pd 


from sklearn.preprocessing import StandardScaler 
from src.myproject.pipelines.prediction_pipeline import CustomData , PredictPipeline


application =  Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age = float(request.form.get('age')),
            bp = float(request.form.get('bp')),
            bgr = float(request.form.get('bgr')),
            bu = request.form.get('bu'),
            hemo = float(request.form.get('hemo')),
            htn = request.form.get('htn')
        )
        pred_df = data.get_data_as_dataframe()
        logging.info(f"Input DataFrame for prediction: {pred_df}")

        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        logging.info(f"Prediction result: {results[0]}")
        return render_template('home.html', results=results[0])

    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug = True)
