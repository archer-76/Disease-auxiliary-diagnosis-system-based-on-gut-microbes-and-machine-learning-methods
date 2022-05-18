from flask import Flask
from flask import jsonify
from flask import request
import json


@app.route("/flask/login", methods=['POST'])
def login():
    data = request.get_data()
    data = json.loads(data)
    username = data['username']
    password = data['password']

    return jsonify({"login": Login.login(username, password)})  # 返回布尔值
