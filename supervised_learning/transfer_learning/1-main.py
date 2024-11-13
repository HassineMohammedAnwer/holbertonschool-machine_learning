#!/usr/bin/env python3

from tensorflow import keras as K
preprocess_data = __import__('1-transfer_fine-tune').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('fine_tuned_cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)