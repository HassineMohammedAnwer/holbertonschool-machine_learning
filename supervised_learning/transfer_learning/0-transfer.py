#!/usr/bin/env python3
"""transfer.py"""


import os
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def preprocess_data(X, Y):
    """function preprocess data"""
    X = X / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y

(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train, y_tain = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
base_model = K.applications.DenseNet121(weights='imagenet', include_top=False,
                                     input_shape=(224, 224, 3))
base_model.trainable = False
inputs = K.Input(shape=(32, 32, 3))
input = K.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(inputs)
x = base_model(input, training=False)
x = K.layers.GlobalAveragePooling2D()(x)
x = K.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = K.layers.Dropout(0.3)(x)
x = K.layers.BatchNormalization()(x)
x = K.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = K.layers.Dropout(0.3)(x)
x = K.layers.BatchNormalization()(x)
outputs = K.layers.Dense(10, activation='softmax')(x)
model = K.Model(inputs, outputs)
model.compile(optimizer=K.optimizers.Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=500, verbose=1, validation_data=(x_test, y_test))
model.save('tmp_cifar10.h5')