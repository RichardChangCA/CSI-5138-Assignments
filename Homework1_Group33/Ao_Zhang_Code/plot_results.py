"""
CSI 5138:           Assignment 1
Student Name:       Ao Zhang
Student Number:     0300039680
Student Email:      azhan085@uottawa.ca
"""
##### for plotting through X11 #####
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np

def Plotting(current_test, regularization = False):
    """
    Function:
        Plot the result according to different args.
    """
    """
    3 modes, for 3 questions respectively:
        1. "test_N"
        2. "test_d"
        3. "test_sigma"
    1 argument:
        regularization = True or False
    """
    # set a switch case for switch the questions quickly
    switcher = {"test_N": 0,
                "test_d": 1,
                "test_sigma": 2}
    test_num = switcher[current_test]

    # set the value according to the questions for x axis of the plots
    N_all = np.array([2, 5, 10, 20, 50, 100, 200])
    d_all = np.arange(21)
    sigma_all = np.array([0.01, 0.1, 1])

    # prepare the name of regularization or non-regularization
    if regularization:
        reg = "regularized"
    else:
        reg = "noreg"

    # define the x label, y label and title with different modes
    if test_num == 0:
        N = "all"
        d = 5
        sigma = 0.1
        leng = 7
        x = N_all
        title = "E_in, E_out, E_bias of different dataset sizes" + "(" + reg + ")"
        label_x = "number of points in the dataset"
        label_y = "Error Value"
    elif test_num == 1:
        N = 50
        d = "all"
        sigma = 0.1
        leng = 21
        x = d_all
        title = "E_in, E_out, E_bias of different model complexities" + "(" + reg + ")"
        label_x = "order of polynomial model"
        label_y = "Error Value"
    else:
        N = 200
        d = 10
        sigma = "all"
        leng = 3
        x = sigma_all
        title = "E_in, E_out, E_bias of different dataset noise variances" + "(" + reg + ")"
        label_x = "variance of dataset points"
        label_y = "Error Value"

    # load the numpy array produced by the assignment1_tf.py
    E_in = np.load("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + "_" + reg + "_Ein.npy")
    E_out = np.load("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + "_" + reg + "_Eout.npy")
    E_bias = np.load("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + "_" + reg + "_Ebias.npy")

    # plot the figure and save it
    fig = plt.figure(figsize = (8, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    ax1.set_title(title)
    E_in_line, = ax1.plot(x, E_in, "b--")
    E_out_line, = ax1.plot(x, E_out, "r-")
    E_bias_line, = ax1.plot(x, E_bias, "g:")
    ax1.legend([E_in_line, E_out_line, E_bias_line], ["E_in", "E_out", "E_bias"])
    plt.savefig("figures/" + current_test + "_" + reg + ".png")
    plt.show()

if __name__== "__main__":
    """
    3 modes, for 3 questions respectively:
        1. "test_N"
        2. "test_d"
        3. "test_sigma"
    """
    # choose a plot
    current_test = "test_d"
    regularization = True

    Plotting(current_test, regularization)
