from flask import Flask, request, render_template
from src.myproject.logger import logging
from sklearn.preprocessing import LabelEncoder
# Set up logging


import numpy as np 
import pandas as pd 


from sklearn.preprocessing import StandardScaler 
from src.myproject.pipelines.prediction_pipeline import CustomData , PredictPipeline


application =  Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html', results=None)
    else:
        try:
            logging.info("Prediction has been initiated")
            logging.info(f"Received input data: {request.form}")
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

            # Decode the prediction result to human readable format
            label_encoder = LabelEncoder()
            label_encoder.fit(['notckd', 'ckd'])  # Ensure the classes are set correctly
            decoded_result = label_encoder.inverse_transform(results)
            
            logging.info(f"Prediction result: {decoded_result[0]}")
            return render_template('home.html', results=decoded_result[0])
        except Exception as e:
            logging.error(f"Error occurred during prediction: {str(e)}")
            return render_template('home.html', results='Error occurred during prediction')


    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug = True)
