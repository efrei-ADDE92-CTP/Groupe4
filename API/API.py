from flask import Flask,request
import joblib
import json

app = Flask(__name__)

@app.before_first_request
def load_model():
    global model
    model = joblib.load("classifier.joblib")
    print("model loaded")

@app.route('/predict', methods=['POST'])
def predictions():
    data = json.loads(request.data)
    sepal_length = data["sepal length"]
    sepal_width = data["sepal width"]
    petal_length = data["petal length"]
    petal_width = data["petal width"]
    prediction = model.predict([[sepal_length, sepal_width,petal_length,petal_width]])
    labels = ['setosa', 'versicolor', 'virginica']
    return {"prediction" : str(labels[prediction[0]])}

app.run("0.0.0.0",port = 80, debug = True)