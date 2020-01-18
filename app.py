import os
import base64
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json

import model
app = Flask(__name__)
UPLOAD_FOLDER='static'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results",methods=['GET','POST'])
def results():
    if request.method == 'POST':
        print(request)
        # image_string = request.files['file']
        data = request.json
        print(data)
        image_string = json.loads(data)['file']
        print(image_string)
        # filename = secure_filename(upload_file.filename)
        # upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        # image = np.array(Image.open(os.path.join(UPLOAD_FOLDER, filename)))
        image_bytes = bytes(image_string, encoding='utf-8')
        image = np.array(Image.frombytes('F', (28, 28), image_bytes, 'raw'))
        # image_decoded = base64.decodebytes(image_string)
        # image = np.frombuffer(image_decoded, dtype=np.float64)
        output = model.predict(image)
        print(output)
        return jsonify(output)
    return "Error"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)