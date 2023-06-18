from strategies import *
from flask import Flask, request
import flask
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/stockData', methods=["GET", "POST"])
def stockData():
    print("stockData endpoint reached...")
    if request.method == "GET":
        with open("stockData.json", "r") as f:
            data = json.load(f)
            data.append({
                "stockName": "COF",
                "strategies": [True]
            })

            return flask.jsonify(data)
    if request.method == "POST":
        received_data = request.get_json()
        print(f"received data: {received_data}")
        message = received_data['data']
        return_data = {
            "status": "success",
            "message": f"received: {message}"
        }
        return flask.Response(response=json.dumps(return_data), status=201)

if __name__ == "__main__":
    app.run("localhost", 6969)