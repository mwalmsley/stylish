import logging
import time
import functools

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image

from offline.helpers import *




def main(input_image=None):  # an np.array

    mpl.rcParams['figure.figsize'] = (12,12)
    mpl.rcParams['axes.grid'] = False

    content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

    # https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
    # style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')  
    #style_path = 'style-imgs/fire_australia.jpg'
    style_path = 'style-imgs/lightning.jpg'
  
    # print(input_image.min(), input_image.mean(), input_image.max())
    if input_image is not None:
      content_image = tf.constant(np.expand_dims(input_image / input_image.max(), axis=0), dtype=tf.float32)  # this is used for the content targets (via VGG)
    else:
      content_image = load_img(content_path)

    style_image = load_img(style_path)  # used for the style targets (via VGG)

    content_layers = ['block5_conv2']

    # Style layer of interest ***
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)


    """Debugging"""
    # print('content')
    # print(content_image.numpy().min(), content_image.numpy().mean(), content_image.numpy().max())
    # print('style')
    # print(style_image.numpy().min(), style_image.numpy().mean()), style_image.numpy().max()

    # content_arr = np.squeeze((255 * content_image.numpy() / content_image.numpy().max()).astype(np.uint8))
    # print(content_arr)
    # print(content_arr.shape)
    # Image.fromarray(content_arr).save('static/latest_content.jpg')

    # style_arr = np.squeeze(255.*style_image.numpy()/style_image.numpy().max()).astype(np.uint8)
    # print('style', style_arr)
    # Image.fromarray(style_arr).save('static/latest_style.jpg')
    """End debugging"""

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # having gotten the style and content targets, we now optimise the image values (starting from the content)
    image = tf.Variable(content_image)


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


    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(clip_0_1(image))


    ### defining the model
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight=1e-2
    content_weight=1e4

    start = time.time()

    epochs = 10
    steps_per_epoch = 15

    step = 0
    for n in range(epochs):
      for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
      print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    return tensor_to_image(image)


if __name__ == '__main__':
    main()
