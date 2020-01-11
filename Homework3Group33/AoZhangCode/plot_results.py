"""
-----------------------------------------------------------------
CSI 5138: Assignment 3
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
Result plotting code.

The code is for reading the result saved from tensorboard, and
re-orgnizing the plots.
-----------------------------------------------------------------
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# all csv file names
vanilla_20_acc_file = "results_csv/run_test_vanilla_20-tag-accuracy.csv"
vanilla_50_acc_file = "results_csv/run_test_vanilla_50-tag-accuracy.csv"
vanilla_100_acc_file = "results_csv/run_test_vanilla_100-tag-accuracy.csv"
vanilla_200_acc_file = "results_csv/run_test_vanilla_200-tag-accuracy.csv"
vanilla_500_acc_file = "results_csv/run_test_vanilla_500-tag-accuracy.csv"

lstm_20_acc_file = "results_csv/run_test_lstm_20-tag-accuracy.csv"
lstm_50_acc_file = "results_csv/run_test_lstm_50-tag-accuracy.csv"
lstm_100_acc_file = "results_csv/run_test_lstm_100-tag-accuracy.csv"
lstm_200_acc_file = "results_csv/run_test_lstm_200-tag-accuracy.csv"
lstm_500_acc_file = "results_csv/run_test_lstm_500-tag-accuracy.csv"

vanilla_20_loss_file = "results_csv/run_train_lstm_20-tag-loss.csv"
vanilla_50_loss_file = "results_csv/run_train_lstm_50-tag-loss.csv"
vanilla_100_loss_file = "results_csv/run_train_lstm_100-tag-loss.csv"
vanilla_200_loss_file = "results_csv/run_train_lstm_200-tag-loss.csv"
vanilla_500_loss_file = "results_csv/run_train_lstm_500-tag-loss.csv"

lstm_20_loss_file = "results_csv/run_train_vanilla_20-tag-loss.csv"
lstm_50_loss_file = "results_csv/run_train_vanilla_50-tag-loss.csv"
lstm_100_loss_file = "results_csv/run_train_vanilla_100-tag-loss.csv"
lstm_200_loss_file = "results_csv/run_train_vanilla_200-tag-loss.csv"
lstm_500_loss_file = "results_csv/run_train_vanilla_500-tag-loss.csv"

def ReadCsv(filename):
    """
    Function:
        Read the csv file saved from tensorboard.
    """
    x = []
    y = []

    with open(filename) as f:
        csv_reader = list(csv.reader(f, delimiter=','))
        for i in range(len(csv_reader)):
            if i == 0:
                continue
            else:
                current_line = csv_reader[i]
                x.append(int((current_line[1])))
                y.append(float((current_line[2])))

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

def PlotAndSave(v1, v2, v3, v4, v5, xlim, ylim, label1, label2, label3, label4, label5,
                title, xlabel, ylabel, figname):
    """
    Function:
        plot the curve and save it.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # original curve
    if v3 is None:
        x1, y1 = v1
        x2, y2 = v2
        ax1.plot(x1, y1, 'r-.', alpha = 0.5)
        ax1.plot(x2, y2, 'b-.', alpha = 0.5)
        # smoothing
        x1, y1 = SparseData(x1, y1)
        x2, y2 = SparseData(x2, y2)
        # after smooth
        line1, = ax1.plot(x1, y1, 'r-')
        line1.set_label(label1)
        line2, = ax1.plot(x2, y2, 'b-')
        line2.set_label(label2)
        # plot and save
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_title(title)
        ax1.set_xlabel(xlabel + "(unit: steps)")
        ax1.set_ylabel(ylabel)
        ax1.legend()
        ax1.grid()
        plt.savefig("result_images/" + figname + ".png")
        plt.show()
    else:
        x1, y1 = v1
        x2, y2 = v2
        x3, y3 = v3
        x4, y4 = v4
        x5, y5 = v5
        ax1.plot(x1, y1, 'r-.', alpha = 0.5)
        ax1.plot(x2, y2, 'b-.', alpha = 0.5)
        ax1.plot(x3, y3, 'm-.', alpha = 0.5)
        ax1.plot(x4, y4, 'c-.', alpha = 0.5)
        ax1.plot(x5, y5, 'g-.', alpha = 0.5)
        # smoothing
        x1, y1 = SparseData(x1, y1)
        x2, y2 = SparseData(x2, y2)
        x3, y3 = SparseData(x3, y3)
        x4, y4 = SparseData(x4, y4)
        x5, y5 = SparseData(x5, y5)
        # after smooth
        line1,  = ax1.plot(x1, y1, 'r-')
        line1.set_label(label1)
        line2,  = ax1.plot(x2, y2, 'b-')
        line2.set_label(label2)
        line3,  = ax1.plot(x3, y3, 'm-')
        line3.set_label(label3)
        line4,  = ax1.plot(x4, y4, 'c-')
        line4.set_label(label4)
        line5,  = ax1.plot(x5, y5, 'g-')
        line5.set_label(label5)
        # plot and save
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_title(title)
        ax1.set_xlabel(xlabel + "(unit: steps)")
        ax1.set_ylabel(ylabel)
        ax1.legend()
        ax1.grid()
        plt.savefig("result_images/" + figname + ".png")
        plt.show()

############################################################
####################### accuracy data ######################
############################################################
vanilla_20_acc = ReadCsv(vanilla_20_acc_file)
vanilla_50_acc = ReadCsv(vanilla_50_acc_file)
vanilla_100_acc = ReadCsv(vanilla_100_acc_file)
vanilla_200_acc = ReadCsv(vanilla_200_acc_file)
vanilla_500_acc = ReadCsv(vanilla_500_acc_file)

lstm_20_acc = ReadCsv(lstm_20_acc_file)
lstm_50_acc = ReadCsv(lstm_50_acc_file)
lstm_100_acc = ReadCsv(lstm_100_acc_file)
lstm_200_acc = ReadCsv(lstm_200_acc_file)
lstm_500_acc = ReadCsv(lstm_500_acc_file)

############################################################
#################### loss function data ####################
############################################################
vanilla_20_loss = ReadCsv(vanilla_20_loss_file)
vanilla_50_loss = ReadCsv(vanilla_50_loss_file)
vanilla_100_loss = ReadCsv(vanilla_100_loss_file)
vanilla_200_loss = ReadCsv(vanilla_200_loss_file)
vanilla_500_loss = ReadCsv(vanilla_500_loss_file)

lstm_20_loss = ReadCsv(lstm_20_loss_file)
lstm_50_loss = ReadCsv(lstm_50_loss_file)
lstm_100_loss = ReadCsv(lstm_100_loss_file)
lstm_200_loss = ReadCsv(lstm_200_loss_file)
lstm_500_loss = ReadCsv(lstm_500_loss_file)


mode = 0

# print all final accuracy values
if mode == 0:
    vanilla_acc_results = np.array([vanilla_20_acc[1][-1], vanilla_50_acc[1][-1], vanilla_100_acc[1][-1], vanilla_200_acc[1][-1], vanilla_500_acc[1][-1]])
    lstm_acc_results = np.array([lstm_20_acc[1][-1], lstm_50_acc[1][-1], lstm_100_acc[1][-1], lstm_200_acc[1][-1], lstm_500_acc[1][-1]])
    states = np.array([20, 50, 100, 200, 500])
    print(vanilla_acc_results)
    print(lstm_acc_results)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(states, vanilla_acc_results, "r")
    ax1.set_xlabel("State dimensions")
    ax1.set_ylabel("Accuracy on test set")
    ax1.set_title("Vanilla RNN with different state dimensions.")
    plt.savefig("result_images/vanilla_final_acc.png")
    plt.cla()
    ax1.clear()
    ax1.plot(states, lstm_acc_results, "b")
    ax1.set_xlabel("State dimensions")
    ax1.set_ylabel("Accuracy on test set")
    ax1.set_title("LSTM with different state dimensions.")
    plt.savefig("result_images/lstm_final_acc.png")
    plt.show()

# plotting under different modes
if mode == 1:
    PlotAndSave(vanilla_20_acc, vanilla_50_acc, vanilla_100_acc, vanilla_200_acc, vanilla_500_acc,
                [0, 4000], [0.78, 0.88], 
                "vanilla_20", "vanilla_50", "vanilla_100", "vanilla_200", "vanilla_500",
                "accuracies of different state sizes of vanilla RNN", "training steps", "accuracy", 
                "vanilla_acc")
if mode == 2:
    PlotAndSave(vanilla_20_loss, vanilla_50_loss, vanilla_100_loss, vanilla_200_loss, vanilla_500_loss,
                [0, 4000], [0.2, 0.6], 
                "vanilla_20", "vanilla_50", "vanilla_100", "vanilla_200", "vanilla_500",
                "loss values of different state sizes of vanilla RNN", "training steps", "loss value", 
                "vanilla_loss")
if mode == 3:
    PlotAndSave(lstm_20_acc, lstm_50_acc, lstm_100_acc, lstm_200_acc, lstm_500_acc,
                [0, 4000], [0.8, 0.88], 
                "lstm_20", "lstm_50", "lstm_100", "lstm_200", "lstm_500",
                "accuracies of different state sizes of LSTM", "training steps", "accuracy", 
                "lstm_acc")
if mode == 4:
    PlotAndSave(lstm_20_loss, lstm_50_loss, lstm_100_loss, lstm_200_loss, lstm_500_loss,
                [0, 4000], [0.25, 0.6], 
                "lstm_20", "lstm_50", "lstm_100", "lstm_200", "lstm_500",
                "loss values of different state sizes of LSTM", "training steps", "loss value", 
                "lstm_loss")
if mode == 5:
    PlotAndSave(vanilla_50_acc, lstm_50_acc, None, None, None,
                [0, 4000], [0.8, 0.88], 
                "vanilla", "lstm", None, None, None,
                "accuracies of vanilla RNN and LSTM", "training steps", "accuracy", 
                "compare_acc")
if mode == 6:
    PlotAndSave(vanilla_50_loss, lstm_50_loss, None, None, None,
                [0, 4000], [0.2, 0.7], 
                "vanilla", "lstm", None, None, None,
                "loss values of vanilla RNN and LSTM", "training steps", "loss value", 
                "compare_loss")