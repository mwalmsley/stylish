import tensorflow as tf 
import numpy as np

import offline


def load_model():
    return tf.keras.models.load_model('static/mnist')


def predict(image):

    model = load_model()

    image_size=28
    num_channels=1

    # image_array = []
    # # image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    # image_array.append(image)
    # image_array = np.array(image_array, dtype=np.uint8)
    # image_array = image_array.astype('float32')
    # image_array = image_array / 256.  # TODO refactor into normalise/denormalise
    # x_batch = image_array  # ignore lint error

    x_batch = image.reshape(1, image_size, image_size, num_channels) 
    print(x_batch.shape)

    y_batch = model.predict(x_batch)
    print(y_batch)

    out = {
            'prediction': int(np.argmax(y_batch[0]))
        }

    return out


if __name__ == '__main__':
    # for debugging
    (train_images, _), _ = offline.get_data()
    image = train_images[0]
    print(image.shape)
    result = predict(image)
    print(result)
