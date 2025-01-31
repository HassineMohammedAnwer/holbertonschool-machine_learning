#!/usr/bin/env python3
"""
    Define Simple GAN class
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """sdfvze"""
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        self.learning_rate    = learning_rate
        self.beta_1=.5
        self.beta_2=.9
        # define the generator loss and optimizer:
        self.generator.loss = lambda x : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer, loss=generator.loss )
        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x,y : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """zfvgefzvezfv"""
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """sqdvzfvefbhn"""
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()    
    def train_step(self,useless_argument):
        """ GANs train"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Get real and fake samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)

            gradients = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)

            gen_out = self.discriminator(fake_samples, training=True)
            gen_loss = self.generator.loss(gen_out)

        gen_grad = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
