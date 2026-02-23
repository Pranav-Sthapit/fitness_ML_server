import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from functions import predict_fitness_and_recommend

app=Flask(__name__)
CORS(app)
@app.route('/prediction_and_recommendation',methods=['POST'])
def predict():

    data=request.json

    activity = data["activity"]
    age = data["age"]
    bloodPressure = data["bloodPressure"]
    gender = data["gender"]
    heartRate = data["heartRate"]
    height = data["height"]
    nutrition = data["nutrition"]
    sleepHours = data["sleepHours"]
    smoke = data["smoke"]
    weight = data["weight"]

    fitness,recommendation=predict_fitness_and_recommend(activity,age,bloodPressure,gender,heartRate,height,nutrition,sleepHours,smoke,weight)

    return jsonify({"fitness":fitness,"recommendation":recommendation})
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default 5000 locally
    app.run(host="0.0.0.0", port=port)