from __future__ import absolute_import, division, print_function, unicode_literals
from helpers import *
from feedforward import CNN

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

tfs = tf.summary
tfk = tf.keras


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_image = load_img(content_path)
style_image = load_img(style_path)

## random, remove
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape

content_layers = ['block5_conv2']

# Style layer of interest ***
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image = tf.Variable(content_image)

filters = [64, 64, 3]
kernel_sizes = [9, 9, 9]
strides = [1, 1, 1]

cnn = CNN(filters,
          kernel_sizes,
          strides)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        image = cnn(content_image, training=True)
        image = (1 + tfk.activations.tanh(image)) / 2
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        grad = tape.gradient(loss, cnn.trainable_variables)
        opt.apply_gradients(zip(grad, cnn.trainable_variables))
        # 

        return loss

### defining the model
opt = tf.optimizers.Adam(learning_rate=1e-3, beta_1=0.99, epsilon=1e-1)
style_weight=1e-2
content_weight=1e4

start = time.time()

epochs = 100
steps_per_epoch = 100

step = 0
loss = None

train_summary_writer = tfs.create_file_writer('summary/style-transfer')

with train_summary_writer.as_default():
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            loss = train_step(image)
 
            tfs.scalar('Loss', loss, step=step)
            tfs.image('Image', cnn(image), step=step)

            print(".", end='')
        print("Train step: {}, Loss: {}".format(step, loss))

end = time.time()
print("Total time: {:.1f}".format(end-start))


# total variational loss include
