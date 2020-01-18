import json
import base64
import numpy as np
import requests
import datetime



if __name__ == '__main__':
    # for debugging
    image_loc = 'static/example_mnist_image.txt'

    # run once, otherwise use saved image
    # from offline import get_data
    # (train_images, _), _ = get_data()
    # image = train_images[0].astype(np.float32)
    # np.savetxt(image_loc, np.squeeze(image))
    # exit()

    image = np.expand_dims(np.loadtxt(image_loc), axis=-1)
    start_time = datetime.datetime.now()
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
    
    time_elapsed = datetime.datetime.now() - start_time
    print(time_elapsed.seconds)