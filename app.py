import os
import logging
import base64
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json
# from flask_cors import CORS

import model
app = Flask(__name__)
# CORS(app)

UPLOAD_FOLDER='static'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results",methods=['GET','POST'])
def results():
    if request.method == 'POST':
        logging.debug(request)
        # image_string = request.files['file']
        data = request.json
        logging.debug(data)
        json_payload = json.loads(data)
        image_string = json_payload['image']
        image_size = json_payload['image_size']
        # logging.debug(image_string)
        # image_bytes = bytes(image_string, encoding='utf-8')
        # image = np.array(Image.frombytes('F', (image_size, image_size), image_bytes, 'raw'))
        # output = model.predict(image)
        # logging.info(output)
        # response = jsonify(output)
        response = flask.Response("success!")
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    return "Error"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)