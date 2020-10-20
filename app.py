import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('loan.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    output = model.predict(features_value)
    if(output[0] == 0):
        return render_template('answer1.html')
    else:
        return render_template('answer2.html')


if __name__ == "__main__":
    app.run()
