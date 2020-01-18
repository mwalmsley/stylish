import json
import base64
import numpy as np
import requests

from offline import get_data
# for debugging

if __name__ == '__main__':

    (train_images, _), _ = get_data()
    image = train_images[0].astype(np.float32)

    print('Begin post')
    image_s = base64.b64encode(image)
    # print(s)

    # post_body = json.dumps()
    # print(post_body)

    data = json.dumps({'image': image_s.decode('utf-8'), 'image_size': 28})
    print(data)


    r = requests.post('http://0.0.0.0:5000/results', json=data)
    if r.ok:
        print(r.json())