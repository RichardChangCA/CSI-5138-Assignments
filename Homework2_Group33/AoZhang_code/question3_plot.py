import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# all csv file names
CNN_BN_acc = "csv_data/run-test_CNN_BN-tag-accuracy.csv"
CNN_BN_loss = "csv_data/run-train_CNN_BN-tag-loss.csv"
CNN_dropout_acc = "csv_data/run-test_CNN_dropout-tag-accuracy.csv"
CNN_dropout_loss = "csv_data/run-train_CNN_dropout-tag-loss.csv"
CNN_acc = "csv_data/run-test_CNN-tag-accuracy.csv"
CNN_loss = "csv_data/run-train_CNN-tag-loss.csv"

MLP_BN_acc = "csv_data/run-test_MLP_BN-tag-accuracy.csv"
MLP_BN_loss = "csv_data/run-train_MLP_BN-tag-loss.csv"
MLP_dropout_acc = "csv_data/run-test_MLP_dropout-tag-accuracy.csv"
MLP_dropout_loss = "csv_data/run-train_MLP_dropout-tag-loss.csv"
MLP_acc = "csv_data/run-test_MLP-tag-accuracy.csv"
MLP_loss = "csv_data/run-train_MLP-tag-loss.csv"

softmax_BN_acc = "csv_data/run-test_softmax_BN-tag-accuracy.csv"
softmax_BN_loss = "csv_data/run-train_softmax_BN-tag-loss.csv"
softmax_dropout_acc = "csv_data/run-test_softmax_dropout-tag-accuracy.csv"
softmax_dropout_loss = "csv_data/run-train_softmax_dropout-tag-loss.csv"
softmax_acc = "csv_data/run-test_softmax-tag-accuracy.csv"
softmax_loss = "csv_data/run-train_softmax-tag-loss.csv"


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

def SparseData(x_in, y_in, window = 45, order = 1):
    """
    Function:
        Smooth the plot.
    """
    x = x_in
    y = savgol_filter(y_in, window, order)
    return x, y

def PlotAndSave(v1, v2, v3, xlim, ylim, label1, label2, label3, 
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
        ax1.plot(x1, y1, 'r-.', alpha = 0.6)
        ax1.plot(x2, y2, 'b-.', alpha = 0.6)
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
        ax1.plot(x1, y1, 'r-.', alpha = 0.6)
        ax1.plot(x2, y2, 'b-.', alpha = 0.6)
        ax1.plot(x3, y3, 'g-.', alpha = 0.6)
        # smoothing
        x1, y1 = SparseData(x1, y1)
        x2, y2 = SparseData(x2, y2)
        x3, y3 = SparseData(x3, y3)
        # after smooth
        line,  = ax1.plot(x1, y1, 'r-')
        line.set_label(label1)
        line_BN,  = ax1.plot(x2, y2, 'b-')
        line_BN.set_label(label2)
        line_drop,  = ax1.plot(x3, y3, 'g-')
        line_drop.set_label(label3)
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

########################################################
######################### MLP data #####################
########################################################
# accuracies
MLP_acc = ReadCsv(MLP_acc)
MLP_BN_acc = ReadCsv(MLP_BN_acc)
MLP_drop_acc = ReadCsv(MLP_dropout_acc)
# losses
MLP_los = ReadCsv(MLP_loss)
MLP_BN_los= ReadCsv(MLP_BN_loss)
MLP_drop_los = ReadCsv(MLP_dropout_loss)

########################################################
######################### CNN data #####################
########################################################
# accuracies
CNN_ac= ReadCsv(CNN_acc)
CNN_BN_ac = ReadCsv(CNN_BN_acc)
CNN_drop_ac = ReadCsv(CNN_dropout_acc)
# losses
CNN_los = ReadCsv(CNN_loss)
CNN_BN_los = ReadCsv(CNN_BN_loss)
CNN_drop_los = ReadCsv(CNN_dropout_loss)

########################################################
##################### softmax data #####################
########################################################
# accuracies
softmax_ac = ReadCsv(softmax_acc)
softmax_BN_ac = ReadCsv(softmax_BN_acc)
softmax_drop_ac = ReadCsv(softmax_dropout_acc)
# losses
softmax_los = ReadCsv(softmax_loss)
softmax_BN_los = ReadCsv(softmax_BN_loss)
softmax_drop_los = ReadCsv(softmax_dropout_loss)

mode = 12

if mode == 1:
    PlotAndSave(CNN_ac, MLP_acc, softmax_ac, [0, 2400], [0.7, 1.], 
                "CNN", "MLP", "softmax", 
                "accuracies of 3 models", "training steps", "accuracy", 
                "allmodels_acc")
elif mode == 2:
    PlotAndSave(CNN_los, MLP_los, softmax_los, [0, 2200], [0, 8], 
                "CNN", "MLP", "softmax", 
                "loss function values of 3 models", "training steps", "loss value", 
                "allmodels_loss")
################################# BN ################################
elif mode == 3:
    PlotAndSave(CNN_ac, CNN_BN_ac, None, [0, 1800], [0.96, 0.992], 
                "CNN without BN", "CNN with BN", "...", 
                "influence of Batch Normalization on the performance of CNN", 
                "training steps", "accuracy", "CNN_BN_acc")
elif mode == 4:
    PlotAndSave(CNN_los, CNN_BN_los, None, [0, 1500], [0, 0.3], 
                "CNN without BN", "CNN with BN", "...", 
                "influence of Batch Normalization on the loss value of CNN", 
                "training steps", "loss value", "CNN_BN_loss")
elif mode == 5:
    PlotAndSave(MLP_acc, MLP_BN_acc, None, [0, 2000], [0.87, 0.98], 
                "MLP without BN", "MLP with BN", "...", 
                "influence of Batch Normalization on the performance of MLP", 
                "training steps", "accuracy", "MLP_BN_acc")
elif mode == 6:
    PlotAndSave(MLP_los, MLP_BN_los, None, [0, 1500], [0, 0.6], 
                "MLP without BN", "MLP with BN", "...", 
                "influence of Batch Normalization on the loss value of MLP", 
                "training steps", "loss value", "MLP_BN_loss")
elif mode == 7:
    PlotAndSave(softmax_ac, softmax_BN_ac, None, [0, 2200], [0.8, 0.925], 
                "softmax without BN", "softmax with BN", "...", 
                "influence of Batch Normalization on softmax regression (accuracy)", 
                "training steps", "accuracy", "softmax_BN_acc")
elif mode == 8:
    PlotAndSave(softmax_los, softmax_BN_los, None, [0, 2200], [0, 10], 
                "softmax without BN", "softmax with BN", "...", 
                "influence of Batch Normalization on softmax regression (loss value)", 
                "training steps", "loss value", "softmax_BN_loss")
################################## dropout ################################
elif mode == 9:
    PlotAndSave(CNN_ac, CNN_drop_ac, None, [0, 1500], [0.96, 0.995], 
                "CNN without dropout", "CNN with dropout", "...", 
                "influence of Drop Out on the performance of CNN", 
                "training steps", "accuracy", "CNN_dropout_acc")
elif mode == 10:
    PlotAndSave(CNN_los, CNN_drop_los, None, [0, 1500], [0, 0.3], 
                "CNN without dropout", "CNN with dropout", "...", 
                "influence of Drop Out on the loss value of CNN", 
                "training steps", "loss value", "CNN_dropout_loss")
elif mode == 11:
    PlotAndSave(MLP_acc, MLP_drop_acc, None, [0, 2200], [0.9, 0.98], 
                "MLP without dropout", "MLP with dropout", "...", 
                "influence of Drop Out on the performance of MLP", 
                "training steps", "accuracy", "MLP_dropout_acc")
elif mode == 12:
    PlotAndSave(MLP_los, MLP_drop_los, None, [0, 1500], [0, 0.6], 
                "MLP without dropout", "MLP with dropout", "...", 
                "influence of Drop Out on the loss value of MLP", 
                "training steps", "loss value", "MLP_dropout_loss")
elif mode == 13:
    PlotAndSave(softmax_ac, softmax_drop_ac, None, [0, 2000], [0.82, 0.90], 
                "softmax without dropout", "softmax with dropout", "...", 
                "influence of Drop Out on softmax regression (accuracy)", 
                "training steps", "accuracy", "softmax_dropout_acc")
elif mode == 14:
    PlotAndSave(softmax_los, softmax_drop_los, None, [0, 2200], [0, 10], 
                "softmax without dropout", "softmax with dropout", "...", 
                "influence of Drop Out on softmax regression (loss value)", 
                "training steps", "loss value", "softmax_dropout_loss")
