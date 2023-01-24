from flask import Response, Flask,request
import joblib
import json
from prometheus_client import Counter
from prometheus_client.core import CollectorRegistry
import prometheus_client

app = Flask(__name__)

c = Counter("Classifier_g4_request_count", "Number of requests processed")

@app.before_first_request
def load_model():
    global model
    model = joblib.load("classifier.joblib")
    print("model loaded")

@app.route('/predict', methods=['POST'])
def predictions():
    c.inc()
    data = json.loads(request.data)
    sepal_length = data["sepal length"]
    sepal_width = data["sepal width"]
    petal_length = data["petal length"]
    petal_width = data["petal width"]
    prediction = model.predict([[sepal_length, sepal_width,petal_length,petal_width]])
    labels = ['setosa', 'versicolor', 'virginica']
    return {"prediction" : str(labels[prediction[0]])}

@app.route("/metrics")
def metrics():
    res = []
    res.append(prometheus_client.generate_latest(c))
    return Response(res, mimetype="text/plain")

app.run("0.0.0.0",port = 80, debug = True)