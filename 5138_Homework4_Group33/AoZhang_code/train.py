"""
-----------------------------------------------------------------
CSI 5138: Assignment 4
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
Training code for the assignment 4.

The code is for training the models. The training process is built
as the requirements of our assignment description. Note that the 
TensorFlow dependency of this code is:            TensorFlow 2.0
-----------------------------------------------------------------
"""
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
##### 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
import time
import pickle
from tqdm import tqdm
from mlxtend.data import loadlocal_mnist
# import models
from vae import vae
from gan import gan
from wgan import wgan

############################## MNIST LOADING ############################
def TranslateLables(labels, num_class):
    """
    Function:
        Transfer ground truth labels to one hot format, e.g.
        1       ->      [0, 1, 0, 0, 0, ..., 0]
    """
    hm_labels = len(labels)
    label_onehot = np.zeros((hm_labels, num_class), dtype = np.float32)
    for i in range(hm_labels):
        current_class = labels[i]
        label_onehot[i, current_class] = 1.
    return label_onehot

def GetMnistData(data_path):
    """
    Function:
        Read mnist dataset and transfer it into wanted format.
        For input:
            if not CNN: (60000, 784)
            elif CNN:   (60000, 28, 28, 1)
        For output: one hot
            [0, 1, 0, 0, ..., 0]
    """
    # read dataset
    X_train, Y_train_original = loadlocal_mnist(
            images_path=data_path + "train-images-idx3-ubyte", 
            labels_path=data_path + "train-labels-idx1-ubyte")
    X_test, Y_test_original = loadlocal_mnist(
            images_path=data_path + "t10k-images-idx3-ubyte", 
            labels_path=data_path + "t10k-labels-idx1-ubyte")
    # transfer into float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255.
    X_test /= 255.

    # find how many classes
    all_classes = np.unique(Y_train_original)
    num_class = len(all_classes) 
    num_input = X_train.shape[1] 

    # transfer label format
    Y_train = TranslateLables(Y_train_original, num_class)
    Y_test = TranslateLables(Y_test_original, num_class)
    return X_train, Y_train, X_test, Y_test, num_input, num_class

############################## CIFAR LOADING ############################
def ReadCifarLabels(cfar_dir):
    """
    Function:
        Read all label names from given cifar files
    """
    file_name = cfar_dir + "batches.meta"
    with open(file_name, 'rb') as fo:
        cfar_dict = pickle.load(fo, encoding='bytes')

    label_names = cfar_dict[b'label_names']
    num_vis = cfar_dict[b'num_vis']
    num_cases_per_batch = cfar_dict[b'num_cases_per_batch']
    return label_names

def ReadCifarData(file_name):
    """
    Function:
        Read all training data from given cifar files
    """
    with open(file_name, 'rb') as fo:
        cfar_dict = pickle.load(fo, encoding='bytes')
    data = cfar_dict[b'data']
    filenames = cfar_dict[b'filenames']
    labels = cfar_dict[b'labels']
    batch_label = cfar_dict[b'batch_label']
    return data, labels, data.shape[1]

def LoadAllCifarData(cfar_dir):
    """
    Function:
        Read all training and testing data from the given cifar directory.
    """
    # all files in the cifar directory
    batch_file_1 = cfar_dir + "data_batch_1"
    batch_file_2 = cfar_dir + "data_batch_2"
    batch_file_3 = cfar_dir + "data_batch_3"
    batch_file_4 = cfar_dir + "data_batch_4"
    batch_file_5 = cfar_dir + "data_batch_5"
    batch_file_test = cfar_dir + "test_batch"
    # concatenate all file names into a list
    all_batch_train = [batch_file_1, batch_file_2, batch_file_3, batch_file_4, batch_file_5]
    X_train = []
    Y_train = []
    # read each file and transfer the data into numpy array
    for each_file in all_batch_train:
        data, labels, input_size = ReadCifarData(each_file)
        X_train.append(data)
        Y_train.append(labels)
    # concatenate all training data and testing data (normalize it into [0, 1])
    X_train = np.concatenate(X_train, axis = 0).astype(np.float32) / 255.
    Y_train = np.concatenate(Y_train, axis = 0).astype(np.float32)
    X_test, Y_test, input_size = ReadCifarData(batch_file_test)
    X_test = X_test.astype(np.float32) / 255.
    Y_test = np.array(Y_test)
    Y_test = Y_test.astype(np.float32)
    return X_train, Y_train, X_test, Y_test, input_size

def TransferToImage(data):
    """
    Function:
        Since the format of CIFAR data is arranged into R, G, B
    separately. Here, we transfer the format into the standard format
    in order to visualize the data.
    """
    r_channel = data[:, :1024].reshape((-1, 32, 32, 1))
    g_channel = data[:, 1024:2048].reshape((-1, 32, 32, 1))
    b_channel = data[:, 2048:].reshape((-1, 32, 32, 1))
    imgs = np.concatenate([r_channel, g_channel, b_channel], axis = -1)
    return imgs

def FormSamples(dataset_name, current_samples):
    """
    Function:
        Reshape the output samples for visualization
    """
    if dataset_name == "MNIST":
        current_samples = current_samples.reshape((10, 20, 28, 28))
    else:
        current_samples = current_samples.reshape((10, 20, 32, 32, 3))

    all_imgs = []
    for i in range(3):
        row_imgs = []
        for j in range(10):
            row_imgs.append(current_samples[i, j])
        row_imgs = np.concatenate(row_imgs, axis = 1)
        all_imgs.append(row_imgs)
    all_imgs = np.concatenate(all_imgs, axis = 0)
    return all_imgs

############################## MAIN FUNCTION ############################
def train(model_name, dataset_name, num_hidden, latent_size, if_plot=False, if_save=True):
    # reset graph
    tf.keras.backend.clear_session()
    # get input according to the dataset name
    if dataset_name == "MNIST":
        Mnist_local_path = "mnist/"
        X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path)
        X_train = X_train.reshape([-1, 28, 28, 1])
    elif dataset_name == "CIFAR":
        cifar_dir = "cifar-10-batches-py/"
        X_train, Y_train, X_test, Y_test, input_size = LoadAllCifarData(cifar_dir)
        X_train = TransferToImage(X_train)
        print(X_train.shape)
    else:
        raise ValueError("Please input the right dataset name.")

    # parameters settings
    if dataset_name == "CIFAR":
        input_size = (32, 32, 3)
    else:
        input_size = (28, 28, 1)

    # training hyper-parameters settings
    batch_size = 256
    epochs = 1000
    hm_batches_train = len(X_train) // batch_size
    hidden_layer_size = 256 # feel free to tune
    sample_size = 600

    # get model according to different model names
    if model_name == "VAE":
        model = vae(input_size, num_hidden, hidden_layer_size, latent_size, batch_size, dataset_name)
    elif model_name == "GAN":
        model = gan(input_size, num_hidden, hidden_layer_size, latent_size, batch_size, dataset_name)
    elif model_name == "WGAN":
        model = wgan(input_size, num_hidden, hidden_layer_size, latent_size, batch_size, dataset_name)
    else:
        raise ValueError("Please input the right model name.")

    # set check points
    checkpoint_dir = './checkpoints/' + model_name + "_" + dataset_name + "_" + \
                str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    if model_name == "VAE":
        checkpoint = tf.train.Checkpoint(optimizer=model.optimizer,
                                        encoder=model.enc,
                                        decoder=model.dec)
    else:
        checkpoint = tf.train.Checkpoint(generator_optimizer=model.gen_optimizer,
                                        discriminator_optimizer=model.disc_optimizer,
                                        generator=model.gen,
                                        discriminator=model.disc)
    
    # set tensorboard
    log_dir = "logs/" + model_name + "_" + dataset_name + "_" + \
            str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size)
    summary_writer = tf.summary.create_file_writer(log_dir)

    if if_plot:
        # live plot for visualizing learning
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    counter = 0
    for epoch_id in tqdm(range(epochs)):
        for each_batch_train in range(hm_batches_train):
            # X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]
            batch_idx = np.random.randint(len(X_train), size=batch_size)
            X_train_batch = X_train[batch_idx]

            # for GAN and WGAN, we use tanh as activation function. Therefore, the input should be
            # transferred into the same format.
            if model_name == "GAN" or model_name == "WGAN":
                X_train_batch = (X_train_batch - 0.5) / 0.5
            
            # start training
            if model_name == "VAE":
                v_loss = model.Training(X_train_batch)
            else:
                g_loss, d_loss = model.Training(X_train_batch)

            # write into tensorboard
            with summary_writer.as_default():
                if model_name == "VAE":
                    tf.summary.scalar(model_name + '_loss', v_loss.numpy(), step=epoch_id*hm_batches_train + each_batch_train)
                else:
                    tf.summary.scalar(model_name + '_G_loss', g_loss.numpy(), step=epoch_id*hm_batches_train + each_batch_train)
                    tf.summary.scalar(model_name + '_D_loss', d_loss.numpy(), step=epoch_id*hm_batches_train + each_batch_train)

            # sampling the results for a specific frequency
            if (epoch_id*batch_size + each_batch_train) % 200 == 0:
                seed = tf.random.normal([sample_size, latent_size])
                # sampling
                if model_name == "VAE":
                    samples = model.Decoding(seed, apply_sigmoid=True)
                else:
                    samples = model.gen(seed, training=False)
                samples = samples.numpy()
                if model_name == "GAN" or model_name == "WGAN":
                    samples = 0.5 * samples + 0.5

                # save the results as numpy arrays
                if if_save:
                    dir_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size)
                    if not os.path.exists(dir_n):
                        os.mkdir(dir_n)
                    file_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size) + \
                                "/" + str(counter) + ".npy"
                    np.save(file_n, samples)
                    counter += 1

                # for visualization during the training (if debug is needed)
                if if_plot:
                    disp_imgs = FormSamples(dataset_name, samples)
                    plt.cla()
                    ax.clear()
                    # ax.imshow(np.squeeze(sample))
                    ax.imshow(disp_imgs)
                    fig.canvas.draw()
                    plt.pause(0.01)

            # save the model as checkpoint.
            if (epoch_id*batch_size + each_batch_train) % 5000 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            


if __name__ == "__main__":
    """
    model names:
        "VAE"
        "GAN"
        "WGAN"
    dataset names:
        "MNIST"
        "CIFAR"
    """
    model_names = ["GAN", "WGAN", "VAE"]

    lantent_sizes = [50]
    num_hiddens = [2]

    for latent_size in lantent_sizes:
        for model_name in model_names:
            train(model_name, "CIFAR", 0, latent_size)

    for num_hidden in num_hiddens:
        for model_name in model_names:
            train(model_name, "CIFAR", num_hidden, 50)
