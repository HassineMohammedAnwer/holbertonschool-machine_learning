#!/usr/bin/env python3
"""4. LeNet-5 (Tensorflow 1)"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ builds a modified version of the LeNet-5 architecture using tensorflow:
    x is a tf.placeholder (m, 28, 28, 1):the input images for the network
    m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot labels for the network
    The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with the he_normal
    __initialization method: tf.keras.initializers.VarianceScaling(scale=2.0)
    All hidden layers requiring activation should use the relu activation function
    you may import tensorflow.compat.v1 as tf
    you may NOT use tf.keras only for the he_normal method.
    Returns:
    a tensor for the softmax activated output
    a training operation that utilizes Adam optimization (with default hyperparameters)
    a tensor for the loss of the netowrk
    a tensor for the accuracy of the network
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer)(x)
    pooling1 = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             kernel_initializer=initializer,
                             activation='relu')(pooling1)
    pooling2 = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)(conv2)
    flat = tf.layers.Flatten()(pooling2)
    f_c1 = tf.layers.Dense(120,
                            activation='relu',
                            kernel_initializer=initializer)(flat)
    f_c2 = tf.layers.Dense(84,
                            activation='relu',
                            kernel_initializer=initializer)(f_c1)
    predicted_output = tf.layers.Dense(10,
                             activation=tf.nn.softmax,
                             kernel_initializer=initializer)(f_c2)
    A = tf.nn.softmax(predicted_output)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=predicted_output)
    train_Adam = tf.train.AdamOptimizer().minimize(loss)
    correct_pred = tf.equal(tf.argmax(predicted_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return A, train_Adam, loss, accuracy
