import tensorflow as tf

from model import get_raw_model

def make_model():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
        
    model = get_raw_model()

    model.fit(train_images, train_labels, epochs=1)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    model.save_weights('static/mnist')

if __name__ == '__main__':
    make_model()
