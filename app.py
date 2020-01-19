import os
import logging
import base64
from flask import Flask, render_template, jsonify, request, Response
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json
from flask_cors import CORS

import model
app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app)

UPLOAD_FOLDER='static'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results",methods=['GET','POST', 'OPTIONS'])
def results():
    if request.method == 'POST' or request.method == 'OPTIONS':
        print('Hello!')
        # logging.debug(request)
        print(request.files)
        image_size = request.files['image_shape']
        print(image_size)
        storage_wrapper = request.files['image']
        print(storage_wrapper)
        image_bytes = storage_wrapper.read()
        print(image_bytes)
        # data = request.json
        # logging.debug(data)
        # json_payload = json.loads(data)
        # image_string = json_payload['image']
        # image_size = json_payload['image_size']
        # logging.debug(image_string)
        # image_bytes = bytes(image_string, encoding='utf-8')
        # image = np.array(Image.frombytes('F', (image_size, image_size), image_bytes, 'raw'))
        print(image)
        print(image.shape)
        # output = model.predict(image)
        # logging.info(output)
        # response = jsonify(output)
        # response = Response("success!")
        response = jsonify({"result": "success!"})
        # response.headers.add('Access-Control-Allow-Origin', '*')
        # response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS') 
        print(response)
        print(response.headers)
        # response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    print('Hello, get!')
    response = jsonify({"result": "get success!"})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    print(response)
    print(response.headers)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)