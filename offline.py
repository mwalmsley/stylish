import tensorflow as tf
import numpy as np


def get_raw_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model


def get_data():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = np.expand_dims(train_images, axis=3)  # add channel dim
    test_images = np.expand_dims(test_images, axis=3)  # add channel dim
    return (train_images, train_labels), (test_images, test_labels)


def make_model():
    model = get_raw_model()

    # TODO change to our problem
    (train_images, train_labels), (test_images, test_labels) = get_data()


    model.fit(train_images, train_labels, epochs=1)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    model.save('static/mnist')

if __name__ == '__main__':
    make_model()
