"""
-----------------------------------------------------------------
CSI 5138: Assignment 4
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
Generated images reading code for the assignment 4.

The code is for reading the generated samples saved during the 
training process.
-----------------------------------------------------------------
"""
import numpy as np
import matplotlib
# for remote plot through X11
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from glob import glob

def ReadSamples(file_name):
    """
    Function:
        Read all stored samples
    """
    samples = np.load(file_name)
    return samples

def ReadRandomFromIds(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size, ids=None):
    """
    Function:
        Combine samples randomly from the same training process.
    """
    # get the right files
    dir_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + \
                str(latent_size) + "_" + str(hidden_layer_size) + "/"

    all_samples = []
    hm_ids = len(ids)
    for file_id in ids:
        # get image sequence number
        file_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + \
                    "_" + str(latent_size) + "_" + str(hidden_layer_size) + "/" + str(file_id) + ".npy"
        current_samples = ReadSamples(file_n)
        # select random images from the index
        current_samples_inds = np.arange(current_samples.shape[0])
        np.random.shuffle(current_samples_inds)
        current_samples = current_samples[current_samples_inds]
        samples_file = current_samples[:2]
        # append all random images into a big list
        all_samples.append(samples_file)
    all_samples = np.concatenate(all_samples, axis = 0)

    if dataset_name == "MNIST":
        all_samples = all_samples.reshape((-1, 20, 28, 28))
    else:
        all_samples = all_samples.reshape((-1, 20, 32, 32, 3))

    all_imgs = []
    for i in range(all_samples.shape[0]):
        row_imgs = []
        for j in range(20):
            row_imgs.append(all_samples[i, j])
        row_imgs = np.concatenate(row_imgs, axis = 1)
        all_imgs.append(row_imgs)
    all_imgs = np.concatenate(all_imgs, axis = 0)

    return all_imgs

def PlotChanges(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size):
    """
    Function:
        Result of inspecting.
    """
    fig = plt.figure(figsize = (18, 5))
    ax1 = fig.add_subplot(111)

    imgs_all = []
    for count in range(5):
        if dataset_name == "MNIST":
            indexes = np.random.randint(count*100, (count+1)*100, size=10)
        else:
            indexes = np.random.randint(2*count*100, (2*count+1)*100, size=10)
        img = ReadRandomFromIds(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size, indexes)
        imgs_all.append(img)
    imgs_all = np.concatenate(imgs_all, axis=0)

    ax1.imshow(imgs_all, cmap="gray")
    ax1.axis("off")
    # ax1.set_title(str(file_id))
    save_image_name = "examples/" + model_name + "_" + dataset_name + "_" + \
        str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size) + ".png"
    plt.savefig(save_image_name)
    plt.show()

def PlotResults(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size):
    """
    Function:
        Result of inspecting.
    """
    fig = plt.figure(figsize = (18, 5))
    ax1 = fig.add_subplot(111)

    if dataset_name == "MNIST":
        indexes = np.random.randint(400, 500, size=50)
    else:
        indexes = np.random.randint(800, 1000, size=50)
    img = ReadRandomFromIds(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size, indexes)
    imgs_all = img

    ax1.imshow(imgs_all, cmap="gray")
    ax1.axis("off")
    # ax1.set_title(str(file_id))
    save_image_name = "examples/" + model_name + "_" + dataset_name + "_" + \
        str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size) + "results.png"
    plt.savefig(save_image_name)
    plt.show()

def ReadAllSamples(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size, ids=None):
    """
    Function:
        Read all files under the current directory
    """
    # get the right files
    dir_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + \
                str(latent_size) + "_" + str(hidden_layer_size) + "/"
    all_files = glob(dir_n + "*.npy")
    num_files = len(all_files)

    fig = plt.figure(figsize = (18, 10))
    ax1 = fig.add_subplot(111)

    for file_id in range(350, num_files):
        # get image sequence number
        file_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + \
                    "_" + str(latent_size) + "_" + str(hidden_layer_size) + "/" + str(file_id) + ".npy"
        current_samples = ReadSamples(file_n)
        if dataset_name == "MNIST":
            current_samples = current_samples.reshape((20, 30, 28, 28))
        else:
            current_samples = current_samples.reshape((20, 30, 32, 32, 3))

        all_imgs = []
        for i in range(10):
            row_imgs = []
            for j in range(10):
                row_imgs.append(current_samples[i, j])
            row_imgs = np.concatenate(row_imgs, axis = 1)
            all_imgs.append(row_imgs)
        all_imgs = np.concatenate(all_imgs, axis = 0)    

        plt.cla()
        ax1.clear()
        ax1.imshow(all_imgs, cmap="gray")
        ax1.axis("off")
        ax1.set_title(str(file_id))
        fig.canvas.draw()
        plt.pause(1)


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
    model_name = "VAE"
    dataset_name = "CIFAR"

    num_hidden = 0
    latent_size = 10
    hidden_layer_size = 256

    # ReadAllSamples(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size)

    PlotChanges(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size)
    PlotResults(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size)