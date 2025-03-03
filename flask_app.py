from flask import Flask , request , render_template
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
        print(pred_df)
         # Check and drop 'classification' column if it exists
        if 'classification' in pred_df.columns:
            pred_df = pred_df.drop(columns=['classification'])
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = results[0])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug = True)
    