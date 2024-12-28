import pickle
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


model_path = os.path.join(os.getcwd(), 'models', 'random_forest_model.pkl')
preprocessor_path = os.path.join(os.getcwd(), 'models', 'preprocessor.pkl')

rm_model = pickle.load(open(model_path, 'rb'))
preprocessor = pickle.load(open(preprocessor_path, 'rb'))

@app.route('/')
def home() :
    return render_template('index.html')

@app.route('/predict',methods = ['POST','GET'])
def predict() :
    if request.method == 'GET' :
        return render_template('index.html')
    else:
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))
        data = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns=['sepal_length','sepal_width','petal_length','petal_width'])
        data = preprocessor.transform(data)
        result = rm_model.predict(data)

        mapping = {0:'Setosa',1:'Versicolor',2:'Virginica'}   

        return render_template('home.html',prediction = mapping[result[0]])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')    