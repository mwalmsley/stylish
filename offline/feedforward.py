import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import tensorflow_addons as tfa

class CNN(tf.keras.Model):

    def __init__(self, filters, kernel_sizes, strides, name='name', **kwargs):

        super(CNN, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides


    def build(self, input_shape):

        zipped = zip(self.filters, self.kernel_sizes, self.strides)

        self.convs = [Conv2D(filters, size, stride, padding='same')
                      for filters, size, stride in zipped]

        self.instance_norms = [tfa.layers.InstanceNormalization(axis=3) \
                               for i in range(len(self.filters) - 1)]


    def call(self, tensor, training=True):

        for conv, instance_norm in zip(self.convs[:-1], self.instance_norms):

            tensor = conv(tensor)

            tensor = instance_norm(tensor)

            tensor = tf.nn.relu(tensor)

        tensor = self.convs[-1](tensor)

        return tensor
