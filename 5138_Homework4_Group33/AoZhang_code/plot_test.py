"""
-----------------------------------------------------------------
CSI 5138: Assignment 4
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
Loss function plotting code for the assignment 4.

The code is for reading the result saved from tensorboard, and
re-orgnizing the plots.
-----------------------------------------------------------------
"""
import numpy as np
import matplotlib
# for remote plot through X11
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv
from glob import glob


def ReadCsv(filename):
    """
    Function:
        Read the csv file saved from tensorboard.
    """
    x = []
    y = []
    # open csv file.
    with open(filename) as f:
        csv_reader = list(csv.reader(f, delimiter=','))
        for i in range(len(csv_reader)):
            if i == 0:
                continue
            else:
                current_line = csv_reader[i]
                x.append(int((current_line[1])))
                y.append(float((current_line[2])))
    # make it into numpy array
    x = np.array(x)
    y = np.array(y) 
    return x, y

def SparseData(x_in, y_in, window = 25, order = 1):
    """
    Function:
        Smooth the plot.
    """
    x = x_in
    y = savgol_filter(y_in, window, order)
    return x, y

def PlotWithSparse(x, y, ax, color, line_label):
    """
    Function:
        Plot and save the figure.
    """
    # two different styles for plotting
    style1 = color + '-.'
    style2 = color + '-'
    ax.plot(x, y, style1, alpha = 0.5)
    x_prime, y_prime = SparseData(x, y)
    line, = ax.plot(x_prime, y_prime, style2)
    # set labels for legend
    line.set_label(line_label)

def GetOnePlot(model_name, dataset_name, num_hidden, latent_size):
    """
    Function:
        Get the tensorboard file with the specific requirements.
    """
    # get the file name
    file_pref = "tensorboard_scv/run-" + model_name + "_" + dataset_name + "_" + str(num_hidden) + \
            "_" + str(latent_size) + "_" + str(256) + "-tag-" + model_name + "_"
    if model_name == "GAN" or model_name == "WGAN":
        file_descrip = "G_loss.csv"
    else:
        file_descrip = "loss.csv"
    file_name = file_pref + file_descrip

    x, y = ReadCsv(file_name)
    return x, y

def PlotAll(mode, model_name, dataset_name, num_hiddens, latent_sizes, xlim, ylim):
    """
    Function:
        Plot all data for comparison.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['r', 'b', 'm', 'g', 'y']

    if mode == "latents":
        assert len(latent_sizes) > 0
        for i in range(len(latent_sizes)):
            latent_size = latent_sizes[i]
            x, y = GetOnePlot(model_name, dataset_name, num_hiddens, latent_size)
            if latent_size == 100:
                x = x[:len(x)//2]
                y = y[:len(y)//2]
            PlotWithSparse(x, y, ax, colors[i], model_name + " with latent size " + str(latent_size))
        ax.set_title("Influen of different latent sizes on " + model_name)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("training steps")
        ax.set_ylabel("loss values")
        ax.legend()
        ax.grid()
        plt.savefig("output_figures/" + model_name + "_" + dataset_name + "_" + "latents.png")
        plt.show()
    else:
        assert len(num_hiddens) > 0
        for i in range(len(num_hiddens)):
            num_hidden = num_hiddens[i]
            x, y = GetOnePlot(model_name, dataset_name, num_hidden, latent_sizes)
            if num_hidden == 0:
                x = x[:len(x)//2]
                y = y[:len(y)//2]
            if dataset_name == "CIFAR":
                PlotWithSparse(x, y, ax, colors[i], model_name + " with hidden layers " + str(num_hidden + 4))
            else:
                PlotWithSparse(x, y, ax, colors[i], model_name + " with hidden layers " + str(num_hidden + 3))
        ax.set_title("Influen of different hidden layers sizes on " + model_name)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("training steps")
        ax.set_ylabel("loss values")
        ax.legend()
        ax.grid()
        plt.savefig("output_figures/" + model_name + "_" + dataset_name + "_" + "hidden.png")
        plt.show()


if __name__ == "__main__":
    """
    model names:
        "VAE"
        "GAN"
        "WGAN"
    dataset names:
        "MNIST"
        "CIFAR"
    mode names:
        "hiddens"
        "latents"
    """
    model_names = ["GAN", "WGAN", "VAE"]
    dataset_names = ["MNIST", "CIFAR"]
    # latent_sizes = [10, 20, 50, 100, 200]
    latent_sizes = 100
    num_hiddens = [0,1,2]
    # num_hiddens = 0

    
    if isinstance(latent_sizes, list):
        mode = "latents"
    else:
        mode = "hiddens"
        
    PlotAll("hiddens", "WGAN", "MNIST", num_hiddens, latent_sizes, [0, 100000], [-0.2,0.15])