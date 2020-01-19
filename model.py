import tensorflow as tf 
import numpy as np
from PIL import Image

from offline.main_proxy import main


def load_model():
    return tf.keras.models.load_model('static/mnist')


def predict(image):

    image.save('static/latest_input.jpg')
    resized_image = image.resize((224, 224))
    image_arr = np.array(resized_image)[:, :, :3].astype(np.float32) # drop alpha channel if it exists
    print(image_arr.shape)
    print(image_arr.min(), image_arr.mean(), image_arr.max())  # why all 0's?
    # model = load_model()

    # image_size=28
    # num_channels=1

    # image_array = []
    # # image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    # image_array.append(image)
    # image_array = np.array(image_array, dtype=np.uint8)
    # image_array = image_array.astype('float32')
    # image_array = image_array / 256.  # TODO refactor into normalise/denormalise
    # x_batch = image_array  # ignore lint error

    # x_batch = image.reshape(1, image_size, image_size, num_channels) 
    # print(x_batch.shape)

    # y_batch = model.predict(x_batch)
    # print(y_batch)

    y = main(image_arr)
    styled_loc = 'static/latest_styled.jpg'
    y.save(styled_loc)

    out = {
            'original_image': 'static/example_content.jpg',
            'styled_image': styled_loc
        }

    return out


if __name__ == '__main__':

    image = Image.open('static/example_content.jpg')
    result = predict(image)
    print(result)
