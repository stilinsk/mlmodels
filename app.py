import pickle
from flask import Flask, request, app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app =Flask(__name__)
regmodel =pickle.load(open('regmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input_data = np.array(list(data.values())).reshape(1, -1)
    output = regmodel.predict(input_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    
    data = [float(x) for x in request.form.values()]
    input_data = np.array(data).reshape(1, -1)
    output = regmodel.predict(input_data)[0]
    return render_template("home.html", prediction_text="THE INSURANCE COST PREDICTION IS {}".format(output))




    
if __name__=="__main__":
    app.run(debug=True)


