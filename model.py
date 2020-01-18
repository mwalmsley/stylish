import tensorflow as tf 
import numpy as np

def get_raw_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model


def predict(graph, image):
    image_size=224
    num_channels=3
    image_array = []

    # image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    image_array.append(image)
    image_array = np.array(image_array, dtype=np.uint8)
    image_array = image_array.astype('float32')
    image_array = image_array / 256.  # TODO refactor into normalise/denormalise
    x_batch = image_array.reshape(1, image_size, image_size, num_channels)

    y_batch = model.predict(x_batch)

    out = {
        "daisy":str(result[0][0]),
        "sunflowers":str(result[0][1]),
        "dandelion":str(result[0][2]),
        "roses":str(result[0][3]),
        "tulips":str(result[0][4])
        }

    return out