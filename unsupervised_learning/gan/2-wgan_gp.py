#!/usr/bin/env python3
"""1. Wasserstein GANs"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """Wasserstein GAN with Gradient Penalty."""
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9
        self.lambda_gp = lambda_gp

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

    # Generator of interpolated samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """Generate interpolated samples."""
        u = tf.random.uniform([self.batch_size, 1])
        return u * real_sample + (1 - u) * fake_sample

    # Compute the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty."""
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        return tf.reduce_mean((norm - 1.0) ** 2)

    # Overloading train_step()
    def train_step(self, useless_argument):
        """Training step for WGAN with gradient penalty."""
        # Train the discriminator multiple times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Get real and fake samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                # Get interpolated samples
                interpolated_samples = self.get_interpolated_sample(
                    real_samples, fake_samples)

                # Compute discriminator outputs
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                # Compute discriminator loss
                discr_loss = self.discriminator.loss(real_output, fake_output)

                # Compute gradient penalty
                gp = self.gradient_penalty(interpolated_samples)

                # Add gradient penalty to discriminator loss
                new_discr_loss = discr_loss + self.lambda_gp * gp

            # Compute gradients and update discriminator weights
            gradients = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables))

        # Train the generator once
        with tf.GradientTape() as tape:
            # Get fake samples
            fake_samples = self.get_fake_sample(training=True)

            # Compute generator output
            gen_output = self.discriminator(fake_samples, training=True)

            # Compute generator loss
            gen_loss = self.generator.loss(gen_output)

        # Compute gradients and update generator weights
        gen_grad = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_grad, self.generator.trainable_variables))

        # Return losses for monitoring
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
