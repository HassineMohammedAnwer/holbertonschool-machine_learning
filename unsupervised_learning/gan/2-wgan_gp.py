#!/usr/bin/env python3
"""1. Wasserstein GANs"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """Wasserstein GAN class with clipping weights"""
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Define the generator loss and optimizer
        self.generator.loss = lambda x: -tf.reduce_mean(x)  # Wasserstein loss
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer, loss=self.generator.loss)

        # Define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.reduce_mean(y) - tf.reduce_mean(x))  # Wasserstein loss
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    # Generator of fake samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples."""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # Generator of real samples of size batch_size
    def get_real_sample(self, size=None):
        """Generate real samples."""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # Overloading train_step()
    def train_step(self, useless_argument):
        """Training step for WGAN."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                interpolated_samples = self.get_interpolated_sample(
                    real_samples, fake_samples)

                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                discr_loss = self.discriminator.loss(real_output, fake_output)

                gp = self.gradient_penalty(interpolated_samples)

                new_discr_loss = discr_loss + self.lambda_gp * gp

            gradients = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:

            fake_samples = self.get_fake_sample(training=True)

            gen_output = self.discriminator(fake_samples, training=True)

            gen_loss = self.generator.loss(gen_output)

        gen_grad = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_grad, self.generator.trainable_variables))

        # Return losses for monitoring
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
