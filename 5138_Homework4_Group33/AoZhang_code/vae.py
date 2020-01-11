"""
-----------------------------------------------------------------
CSI 5138: Assignment 4
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
VAE model code for the assignment 4.

The code is for building the VAE model. Note that the TensorFlow 
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

class vae(tf.keras.Model):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                latent_size, batch_size, dataset_name):
        """
        Function:
            Initialization of all variables
        """
        # build on tf.keras.Model class
        super(vae, self).__init__()
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.w, self.h, self.ch_in = self.input_size
        self.num_layers = num_hidden_layers
        self.latent_size = latent_size
        self.hidden_size = hidden_layer_size
        self.batch_size = batch_size
        # sample size for debug
        self.sample_size = 1
        # kernel size for generator
        self.k_s = 3
        # build models
        self.dec = self.Decoder()
        self.enc = self.Encoder()
        self.initial_learning_rate = 1e-3
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                self.initial_learning_rate,
                                                decay_steps=10000,
                                                decay_rate=0.9,
                                                staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.trainable_var = self.dec.trainable_variables + self.enc.trainable_variables

    def SampleLatent(self, mean, variance):
        """
        Function:
            Distribution Transferring.
        """
        eps = tf.random.normal(shape = tf.shape(mean))
        return mean + tf.exp(variance / 2) * eps

    def Decoder(self,):
        """
        Function:
            Build Decoder for transfer the latent noise back to the image.
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

        model.add(layers.Conv2DTranspose(self.ch_in, (self.k_s, self.k_s), strides=(2, 2), padding='same', use_bias=False))

        return model

    def Encoder(self,):
        """
        Function:
            Encode the image into the latent representation.
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
        model.add(layers.Dense(2 * self.latent_size))

        return model
    
    def Encoding(self, x):
        """
        Function:
            Split the output of encoder into mean and variance.
        """
        mean, logvar = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def Reparameterize(self, mean, logvar):
        """
        Function:
            Reparametrize trick: transfer Gaussian from (0,1) to the encoder space.
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def Decoding(self, z, apply_sigmoid=False):
        """
        Function:
            Get the output logits of the decoder.
        """
        logits = self.dec(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def LogNormalPdf(self, sample, mean, logvar, raxis=1):
        """
        Function:
            Calculate the PDF of Gaussian distribution.
        """
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum( -.5 * ((sample - mean) ** 2. \
                            * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def Loss(self, x):
        """
        Function:
            Define loss based on the equation:
            L = 1/L sum( logpx_z(x | z_j) + logp_z(z_j) - logqz_x(z_j | x) ), which
        is the first loss function defined by our professor on the slide P24.
        """
        mean, logvar = self.Encoding(x)
        z = self.Reparameterize(mean, logvar)
        x_logit = self.Decoding(z)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        # calculate normal distribution separately
        logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
        logpz = self.LogNormalPdf(z, 0., 0.)
        logqz_x = self.LogNormalPdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def Training(self, x):
        """
        Function:
            Training function, based on tf.GradientTape()
        """
        with tf.GradientTape() as tape:
            loss = self.Loss(x)
        gradients = tape.gradient(loss, self.trainable_var)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_var))
        return loss


