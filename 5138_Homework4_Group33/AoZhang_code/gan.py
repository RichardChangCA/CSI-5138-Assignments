"""
-----------------------------------------------------------------
CSI 5138: Assignment 4
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
GAN model code for the assignment 4.

The code is for building the GAN model. Note that the TensorFlow 
dependency of this code is:              TensorFlow 2.0
-----------------------------------------------------------------
"""
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import glob
import numpy as np
from tensorflow.keras import layers
import time

class gan(object):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                latent_size, batch_size, dataset_name):
        """
        Function:
            Initializing all variables.
        """
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.w, self.h, self.ch_in = self.input_size
        self.num_layers = num_hidden_layers
        self.latent_size = latent_size
        self.hidden_size = hidden_layer_size
        self.batch_size = batch_size
        self.sample_size = 1
        # define kernel size of generator
        self.k_s = 4
        # get all models
        self.gen = self.Generator()
        self.disc = self.Discriminator()
        self.initial_learning_rate = 2e-4
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                self.initial_learning_rate,
                                                decay_steps=20000,
                                                decay_rate=0.95,
                                                staircase=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

    def Noise(self,):
        """
        Function:
            Get latent size noise with Gaussian distribution.
        """
        return tf.random.normal([self.batch_size, self.latent_size])

    def Generator(self,):
        """
        Function:
            Build Generator from latent noise.
            Note: the maximum number of hidden layers added is 2.
        """
        model = tf.keras.Sequential()

        # first several layers for CIFAR dataset
        if self.dataset_name == "CIFAR":
            model.add(layers.Dense(self.w//8*self.h//8*self.hidden_size, input_shape=(self.latent_size,), use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

            model.add(layers.Reshape((self.w//8, self.h//8, self.hidden_size)))

            if self.num_layers >= 1:
                model.add(layers.Conv2DTranspose(self.hidden_size, (self.k_s, self.k_s), strides=(1, 1), padding='same', use_bias=False))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.2))

            model.add(layers.Conv2DTranspose(self.hidden_size//2, (self.k_s, self.k_s), strides=(2, 2), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

        # first several layers for MNIST dataset
        else:
            model.add(layers.Dense(self.w//4*self.h//4*self.hidden_size, input_shape=(self.latent_size,), use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

            model.add(layers.Reshape((self.w//4, self.h//4, self.hidden_size)))

            if self.num_layers >= 1:
                model.add(layers.Conv2DTranspose(self.hidden_size//2, (self.k_s, self.k_s), strides=(1, 1), padding='same', use_bias=False))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(self.hidden_size//2, (self.k_s, self.k_s), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        if self.num_layers >= 2:
            model.add(layers.Conv2DTranspose(self.hidden_size//4, (self.k_s, self.k_s), strides=(1, 1), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(self.ch_in, (self.k_s, self.k_s), strides=(2, 2), padding='same', activation='tanh', use_bias=False))

        return model

    def Discriminator(self,):
        """
        Function:
            Build Discriminator for telling the image is ture or fake.
            Note: the maximum number of hidden layers added is 2.
        """
        model = tf.keras.Sequential()
        
        # first several layers for CIFAR dataset
        if self.dataset_name == "CIFAR":
            model.add(layers.Conv2D(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same',
                                            input_shape=[self.w, self.h, self.ch_in]))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

            model.add(layers.Conv2D(self.hidden_size//2, (3, 3), strides=(2, 2), padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

            if self.num_layers >= 1:
                model.add(layers.Conv2D(self.hidden_size//2, (3, 3), strides=(1, 1), padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.2))

            model.add(layers.Conv2D(self.hidden_size//2, (3, 3), strides=(2, 2), padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

        # first several layers for MNIST dataset
        else:
            model.add(layers.Conv2D(self.hidden_size//2, (self.k_s, self.k_s), strides=(2, 2), padding='same',
                                            input_shape=[self.w, self.h, self.ch_in]))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

            if self.num_layers >= 1:
                model.add(layers.Conv2D(self.hidden_size//2, (self.k_s, self.k_s), strides=(1, 1), padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(self.hidden_size, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        if self.num_layers >= 2:
            model.add(layers.Conv2D(self.hidden_size, (3, 3), strides=(1, 1), padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Flatten())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def DiscriminatorLoss(self, real_output, fake_output):
        """
        Function:
            Discriminator loss.
            Note: Since we want to transfer maximize to minimize, the loss of the fake images should be
        close to the fake (0). Real images should be to real (1).
        """
        real_loss = self.crossEntropy(tf.ones_like(real_output), real_output)
        fake_loss = self.crossEntropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def GeneratorLoss(self, fake_output):
        """
        Function:
            Generator loss.
            Note: Since the first term has nothing to do with the generator, we only consider the second term
        here.
        """
        return self.crossEntropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def Training(self, images):
        """
        Function:
            Get the loss from Discriminator and Generator separately, then apply the gradient descent on both.
        """
        noise = self.Noise()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generate fake images
            generated_images = self.gen(noise, training=True)
            # get output of generated images and real images
            real_output = self.disc(images, training=True)
            fake_output = self.disc(generated_images, training=True)
            # get loss functions
            gen_loss = self.GeneratorLoss(fake_output)
            disc_loss = self.DiscriminatorLoss(real_output, fake_output)
        # calculate gradient
        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        # apply gradient descent
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc.trainable_variables))

        return gen_loss, disc_loss


