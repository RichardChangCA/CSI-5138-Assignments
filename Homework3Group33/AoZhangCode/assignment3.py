"""
-----------------------------------------------------------------
CSI 5138: Assignment 3
Student:            Ao   Zhang
Student Number:     0300039680

-----------------------------------------------------------------
Main code for the assignment 3.

The code is for reading the vocab and wordvector, transferring 
dataset into np.array, building Vanillar RNN and LSTM, training
them.
-----------------------------------------------------------------
"""
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#### other dependencies #####
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from glob import glob
import tensorflow as tf
from tqdm import tqdm

#####################################################################
# This class is for reading the vocabulary and word vectors saved
# in the current directory.
# 
# Then, save the word vector as numpy array in the directory
# ``dataset_numpy'' with vector indexes pointing to a vocabulary
# python dict.
#
# By doing that, we can easily search the word vector using function
# tf.nn.embedding_lookup()
#####################################################################
class WordVectorAndList:
    def __init__(self, vocab_file, vector_file):
        """
        Function:
            Initialization of all values. Note that when the wordvector
        numpy array exists, read the existing array instead of re-produce
        it again for saving running time.
        """
        self.vocab_file = vocab_file
        self.vector_file = vector_file
        self.vocab_list = self.VocabList(self.vocab_file)
        if os.path.exists("dataset_numpy/wordvector.npy"):
            self.word_vector = np.load("dataset_numpy/wordvector.npy")
        else:
            self.word_vector = self.WordVector(self.vector_file, self.vocab_list)

    def VocabList(self, filename):
        """
        Function:
            Build a vocabulary python dictionary with format as:
        {
            "word1" : index1
            "word2" : index2
            ...
         }
        """
        with open(filename, "r") as f:
            all_lines = f.readlines()
            vocab_dict = {}
            count = 0
            for line in all_lines:
                count += 1
                words = line.split()
                vocab_word = words[0]
                vocab_dict[vocab_word] = count
        return vocab_dict

    def WordVector(self, filename, vocab_list):
        """
        Funtion:
            Read the word vector trained by Glove and transfer it
        into np.array with first axies pointing to the word index.
        """
        with open(filename, "r") as f:
            all_lines = f.readlines()
            output_array = np.zeros((len(vocab_list) + 1, 50), dtype = np.float32)
            for line in all_lines:
                characters = line.split()
                if characters[0] not in vocab_list:
                    continue
                word_ind = vocab_list[characters[0]]
                word_vector = []
                for i in range(1, len(characters)):
                    word_vector.append(float(characters[i]))
                word_vector = np.array(word_vector)
                output_array[word_ind] = word_vector
        if not os.path.isdir("dataset_numpy"):
            os.mkdir("dataset_numpy")
        np.save("dataset_numpy/wordvector.npy", output_array)
        return output_array

#####################################################################
# This part is for finding the best sequence length of the data 
# structure.
#
# Two principles introduced:
#       1. The data info could be expressed mostly with this length.
#       2. The model could be trained efficiently with this length.
#####################################################################
def FindAllSequanceLen(dataset_names, all_length):
    """
    Function:
        Scan the data set and append all file length into one list.
    """
    for file_name in dataset_names:
        with open(file_name, "r") as f:
            line = f.readline()
            words = line.split()
            all_length.append(len(words))
    return all_length

def PlotLenHist(train_pos_files, train_neg_files, test_pos_files, test_neg_files):
    """
    Function:
        Find all length w.r.t each dataset file, then plot the distribution
    of the length.
    """
    all_length = []
    all_length = FindAllSequanceLen(train_pos_files, all_length)
    all_length = FindAllSequanceLen(train_neg_files, all_length)
    all_length = FindAllSequanceLen(test_pos_files, all_length)
    all_length = FindAllSequanceLen(test_neg_files, all_length)
    plt.hist(all_length, bins = 500)
    plt.xlim([0, 1000])
    plt.title("Histogram of sequance length")
    plt.xlabel("length bins")
    plt.ylabel("number of sequences")
    plt.savefig("result_images/lengthdistribution.png")
    plt.show()

#####################################################################
# This part is for reading all the dataset from all dataset files,
# then combine and shuffle them into two sets, namely training set
# and test set.
#####################################################################
def cleanSentences(string):
    """
    Function:
        Clean the sentances with these wierd characters.
    """
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def ReadDatasetAsWordIndex(all_filenames, word_list, len_lim):
    """
    Function:
        Read all dataset files, then split the file into word
    by word. After that, find every word index in the vocabulary
    dictionary and store it into a np.array as the current data
    file.

        This could effectively reduce the memory requirement of
    storing the dataset into RAM.
    """
    all_dataset = []
    # read each file name
    for file_name in all_filenames:
        # initialize the output
        output_dataset = np.zeros((len_lim), dtype = np.int32)
        f = open(file_name, "r")
        # there is always only one line in each file
        line=f.readline()
        # remove unecessary characters
        cleaned_line = cleanSentences(line)
        words = cleaned_line.split()
        for i in range(len(words)):
            if i < len_lim:
                # find word index
                output_dataset[i] = word_list[words[i]]
            else:
                continue
        all_dataset.append(output_dataset)
    all_dataset = np.array(all_dataset)
    return all_dataset

def GetAllData(word_list, train_pos_files, train_neg_files, test_pos_files, test_neg_files, length_limit):
    """
    Function:
        Transfer the dataset into np.array for all dataset files
    """
    train_pos_set = ReadDatasetAsWordIndex(train_pos_files, word_list, length_limit)
    train_neg_set = ReadDatasetAsWordIndex(train_neg_files, word_list, length_limit)
    test_pos_set = ReadDatasetAsWordIndex(test_pos_files, word_list, length_limit)
    test_neg_set = ReadDatasetAsWordIndex(test_neg_files, word_list, length_limit)
    return train_pos_set, train_neg_set, test_pos_set, test_neg_set

def CreatDataSet(pos_data, neg_data, name_prefix):
    """
    Function:
        Create the labels according to the file names (since only two)
    classes required. Save all data set into np.array
    """
    # get labels according to it's positive or negative
    len_pos_data = len(pos_data)
    len_neg_data = len(neg_data)
    pos_label = np.zeros((len_pos_data, 2), dtype = np.float32)
    # positive [1., 0.]
    pos_label[:, 0] = 1.
    neg_label = np.zeros((len_neg_data, 2), dtype = np.float32)
    # negative [0., 1.]
    neg_label[:, 1] = 1.
    all_dataset = np.concatenate([pos_data, neg_data], axis = 0)
    all_labels = np.concatenate([pos_label, neg_label], axis = 0)
    assert len(all_dataset) == len(all_labels)
    # set the index of both data and label then shuffle them
    indexes = np.arange(len(all_dataset))
    np.random.shuffle(indexes)
    dataset = all_dataset[indexes]
    labels = all_labels[indexes]
    # save as np.array
    np.save("dataset_numpy/" + name_prefix + "_dataset.npy", dataset)
    np.save("dataset_numpy/" +  name_prefix + "_labels.npy", labels)
    return dataset, labels

def GetTrainAndTestSets(word_list, train_pos_files, train_neg_files, test_pos_files, 
                        test_neg_files, length_limit):
    """
    Function:
        If dataset has been saved, then use the saved dataset directly to avoid
    running the saving process again. (time saving)
    """
    existance = os.path.exists("dataset_numpy/training_dataset.npy") and \
                os.path.exists("dataset_numpy/training_labels.npy") and \
                os.path.exists("dataset_numpy/test_dataset.npy") and \
                os.path.exists("dataset_numpy/test_labels.npy")
    if not existance:
        train_pos_set, train_neg_set, test_pos_set, test_neg_set = GetAllData(word_list, train_pos_files, \
                                    train_neg_files, test_pos_files, test_neg_files, length_limit)
        training_set, training_label = CreatDataSet(train_pos_set, train_neg_set, name_prefix = "training")
        test_set, test_label = CreatDataSet(test_pos_set, test_neg_set, name_prefix = "test")
    else:
        training_set = np.load("dataset_numpy/training_dataset.npy")
        training_label = np.load("dataset_numpy/training_labels.npy")
        test_set = np.load("dataset_numpy/test_dataset.npy")
        test_label = np.load("dataset_numpy/test_labels.npy")
    return training_set, training_label, test_set, test_label

#####################################################################
# This class is for building the model, which effectively combined 
# the vanilla RNN and LSTM together since they share a lot of 
# commons.
#####################################################################
class AssignmentRNNModel:
    def __init__(self, length_limit, word_vector, state_size, name):
        """
        Function:
            Initialization.
        """
        # get name, whether ``vanilla'' or ``lstm''
        self.name = name
        # basic parameters
        self.word_vector = word_vector
        self.rnn_size = state_size
        self.length_limit = length_limit
        self.learning_rate_start = 0.001
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        30, 0.96, staircase=True)
        # placeholders for input and output
        self.X_ids = tf.placeholder(tf.int32, [None, self.length_limit])
        self.Y = tf.placeholder(tf.float32, [None, 2])
        # look up the word vector then transfer the input
        self.X = tf.nn.embedding_lookup(self.word_vector, self.X_ids)
        # last layer's parameter
        self.W_out = tf.Variable(tf.random_normal_initializer()([self.rnn_size, 2]))
        self.B_out = tf.Variable(tf.random_normal_initializer()([2]))
        if self.name == "vanilla":
            # Vanillar RNN cell (default activation: tanh)
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
        elif self.name == "lstm":
            # LSTM cell (default activation: tanh)
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        else:
            raise ValueError("wrong input name, please choose one from either vanilla or lstm")
    
    def RnnModel(self):
        """
        Function:
            Build the RNN model.
        """
        outputs, states = tf.nn.dynamic_rnn(self.rnn_cell, self.X, dtype=tf.float32)
        # mean pool all the ouputs
        average_output = tf.reduce_mean(outputs, axis = 1)
        # transfer to the classes
        prediction = tf.matmul(average_output, self.W_out) + self.B_out
        return prediction
    
    def LossFunction(self):
        """
        Function:
            Define loss function as cross entropy after softmax gating function.
        """
        pred = self.RnnModel()
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.Y, logits = pred)
        loss = tf.reduce_mean(loss)
        return loss

    def TrainModel(self):
        """
        Function:
            Define optimization method as tf.train.AdamOptimizer()
        """
        loss = self.LossFunction()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        learning_operation = optimizer.minimize(loss, global_step = self.global_step)
        return learning_operation

    def Accuracy(self):
        """
        Function:
            Define validation as comparing the prediction with groundtruth. Details would
        be, finding the predction class w.r.t the largest probability, then compare with 
        the true labels.
        """
        pred = self.RnnModel()
        pred = tf.nn.softmax(pred)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
        return accuracy


#####################################################################
# Main function, for training and testing the model.
# all hyperparameters are defined here.
#####################################################################
def main(mode, state_size):
    """
    Main:
        Running the training to get the results.
    """
    # vocabulary file and word vector file from Glove.
    vocab_file = "vocab.txt"
    vector_file = "vectors.txt"
    # all file names (Caution: dataset should be in the current dir)
    train_pos_files = glob("aclImdb/train/pos/*.txt")
    train_neg_files = glob("aclImdb/train/neg/*.txt")
    test_pos_files = glob("aclImdb/test/pos/*.txt")
    test_neg_files = glob("aclImdb/test/neg/*.txt")

    # plot the histogram of length
    plot_hist = False

    # sequence length limitation
    length_limit = 300
    # training parameters
    batch_size = 1000
    epoches = 150
    # get vocabulary and wordvector
    words_dict_creater = WordVectorAndList(vocab_file, vector_file)
    word_list = words_dict_creater.vocab_list
    word_vector = words_dict_creater.word_vector

    # plot the histogram of length
    if plot_hist:
        PlotLenHist(train_pos_files, train_neg_files, test_pos_files, test_neg_files)

    # get all datasets
    training_set, training_label, test_set, test_label = GetTrainAndTestSets(word_list, \
            train_pos_files, train_neg_files, test_pos_files, test_neg_files, length_limit)

    # get the how many batches to go.
    num_train_batch = len(training_set) // batch_size
    num_test_batch = len(test_set) // batch_size
    print([num_train_batch, num_test_batch])

    # building the model
    model = AssignmentRNNModel(length_limit, word_vector, state_size, name = mode)

    # define loss value, accuracy
    loss = model.LossFunction()
    accuracy = model.Accuracy()
    train = model.TrainModel()

    # tensorboard settings
    loss_graph_name = "loss"
    acc_graph_name = "accuracy"
    summary_loss = tf.summary.scalar(loss_graph_name, loss)
    streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(accuracy)
    summary_accuracy = tf.summary.scalar(acc_graph_name, streaming_accuracy)

    # initialization
    init = tf.global_variables_initializer()
    # GPU settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summaries_train = 'logs/train/'
        summaries_test = 'logs/test/'
        folder_name = mode + "_" + str(state_size)
        train_writer = tf.summary.FileWriter(summaries_train + folder_name, sess.graph)
        test_writer = tf.summary.FileWriter(summaries_test + folder_name, sess.graph)
        
        sess.run(init)
        for epoch_id in tqdm(range(epoches)):
            for train_batch_id in range(num_train_batch):
                X_train_batch = training_set[train_batch_id*batch_size : (train_batch_id+1)*batch_size]
                Y_train_batch = training_label[train_batch_id*batch_size : (train_batch_id+1)*batch_size]
                _, loss_val, summary_l, steps = sess.run([train, loss, summary_loss, model.global_step], \
                                                                    feed_dict = {model.X_ids : X_train_batch, \
                                                                                model.Y : Y_train_batch})
                train_writer.add_summary(summary_l, steps)

                """
                When GPU memory is not enough
                """
                if train_batch_id % 20 == 0:
                    sess.run(tf.local_variables_initializer())
                    for test_batch_id in range(num_test_batch):
                        X_test_batch = test_set[test_batch_id*batch_size: (test_batch_id+1)*batch_size]
                        Y_test_batch = test_label[test_batch_id*batch_size: (test_batch_id+1)*batch_size]
                        sess.run([streaming_accuracy_update], feed_dict = {model.X_ids : X_test_batch, \
                                                                            model.Y : Y_test_batch})

                    summary_a = sess.run(summary_accuracy)
                    test_writer.add_summary(summary_a, steps)

if __name__ == "__main__":
    """
    2 modes: "vanilla", "lstm"
    """
    mode = "vanilla"
    # requirements as described in the assignment3
    state_size_requirements = [20, 50, 100, 200, 500]
    state_size = state_size_requirements[4]
    main(mode, state_size)



