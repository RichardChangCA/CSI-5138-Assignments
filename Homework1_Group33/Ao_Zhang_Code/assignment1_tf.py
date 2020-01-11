"""
CSI 5138:           Assignment 1
Student Name:       Ao Zhang
Student Number:     0300039680
Student Email:      azhan085@uottawa.ca
"""
##### for plotting through X11 #####
import matplotlib
matplotlib.use("tkagg")
import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

"""
Create the polynomial model with the given parameters
"""
class PolynomialModel:
    def __init__(self, order, batch_size, learning_rate, index, regularization = False):
        self.order = order + 1
        self.index = index
        # build placeholder for inputs and outputs
        self.X = tf.placeholder(tf.float32, [None, 1], "input")
        self.Y = tf.placeholder(tf.float32, [None, 1], "label")
        self.X_poly = self.RebuildInput()
        # prepare the variables for training
        self.Theta = tf.get_variable("parameters" + str(self.index), [self.order, 1],
                                    dtype = tf.float32, initializer = tf.glorot_uniform_initializer())
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        # regularization lambda
        self.reg_lambda = 0.01

    def RebuildInput(self):
        """
        Function:
            read the input X = [x1, ..., xn]^T and transfer it to
            X_poly = [[0, x1, x1^2, ..., x1^d], [[0, x2, x2^2, ..., x2^d], ..., [0, xn, xn^2, ..., xn^d]]

            where, d is the order of the polynomial.
        """
        for i in range(self.order):
            if i == 0:
                X_poly_form = self.X ** i
            else:
                X_poly_form = tf.concat([X_poly_form, (self.X ** i)], axis = -1)
        return X_poly_form
    
    def Polynomial(self):
        """
        Function:
            calculate the prediction using the definition: Y_ = X * theta
        """
        return tf.matmul(self.X_poly, self.Theta)

    def getMSE(self):
        """
        Function:
            calculate the mean square error by the definition: (X * theta - Y) ** 2
        """
        prediction = self.Polynomial()
        mean_square_err = tf.reduce_sum((prediction - self.Y) ** 2) / self.batch_size
        return mean_square_err

    def GradientDescent(self):
        """
        Function:
            design gradient by using the definition:
            if no regularization:
                theta = theta_{old} / batch_size * sum(2 * (y - theta_{old}^T x) * x)
            if regularization:
                theta = theta_{old} + learning_rate / batch_size * sum(2 * (y - theta_{old}^T x) * x) 
                        - 2 * lambda * learning_rate * theta_{old}
        """
        prediction = self.Polynomial()
        learning = tf.add(self.Theta , self.learning_rate * 2 * \
                        tf.matmul(tf.transpose(self.X_poly), (self.Y - prediction)) / \
                        self.batch_size)
        if self.regularization:
            learning = tf.add(learning, -2 * self.learning_rate * self.reg_lambda * self.Theta)
        operator = self.Theta.assign(learning)
        return operator


def getData(num_data, variance):
    """
    Function:
        Produce the dataset according to the function: 
            Y = cos(2 * pi * X) + Z
        
        where, Z belongs to white noise.
    """
    # uniformly select x_data from (0, 1)
    x = np.random.uniform(0., 1., size = (num_data,))
    # produce Y = cos(2 * pi * X) + Z
    y = np.cos(2 * np.pi * x) + np.random.normal(0, variance, size = (num_data,))
    # make the size (N, ) to (N, 1) for convenience
    x = np.expand_dims(x, axis = -1)
    y = np.expand_dims(y, axis = -1)
    return x, y

def fitData(sess, model, train_x, train_y, test_x, test_y, num_train, variance):
    """
    Function:
        Fit the data into the model to train it.
    """
    # add epoches for gradient descent
    epoches = 2000

    # using gradient descent with proper learning_rate and epoches to learn the function
    op = model.GradientDescent()
    for each_epoch in range(epoches):
        # set the maximum batch size to 5
        test = sess.run(op, feed_dict = {model.X : train_x, model.Y : train_y})

        # add a learning rate decay, to make the results converge better
        if each_epoch // 100: 
            model.learning_rate *= 0.96
    
    # get the outputs
    E_in, para = sess.run([model.getMSE(), model.Theta], feed_dict = {model.X : train_x, model.Y : train_y})
    E_out = sess.run(model.getMSE(), feed_dict = {model.X : test_x, model.Y : test_y})

    return E_in, E_out, para

def experiment(sess, order, num_train, variance, learning_rate, ind, debug = False, regularization = False):
    """
    Function:
        Do M times experiments and get E_in, E_out and E_bias out.
    """
    M = 50
    num_bias = 3000
    num_test = 2000

    # initialize variables
    E_in_all = []
    E_out_all = []
    theta_all = []

    # produce training data and test data
    train_x, train_y = getData(num_train, variance)
    test_x, test_y = getData(num_test, variance)
    bias_x, bias_y = getData(num_bias, variance)
    
    # build the model
    model = PolynomialModel(order, num_train, learning_rate, ind, regularization)
    init = tf.initializers.global_variables()

    for _ in tqdm(range(M)):
        # initialize the model
        sess.run(init)
        model.learning_rate = learning_rate
        # training the model
        E_in, E_out, theta = fitData(sess, model, train_x, train_y, 
                                    test_x, test_y, num_train, variance)
        # store the results
        E_in_all.append(E_in)
        E_out_all.append(E_out)
        theta_all.append(theta)
    
    E_in_all = np.array(E_in_all)
    E_out_all = np.array(E_out_all)
    theta_all = np.array(theta_all)

    # calculate E_in_bar, E_out_bar, theta_bar
    E_in_bar = np.mean(E_in_all, axis = 0)
    E_out_bar = np.mean(E_out_all, axis = 0)
    theta_bar = np.mean(theta_all, axis = 0)

    # assign the mean theta to the model variables
    op = model.Theta.assign(theta_bar)
    sess.run(op)

    # calculate E_bias
    E_bias = sess.run(model.getMSE(), feed_dict = {model.X : bias_x, model.Y : bias_y})

    # debug: monitoring the result of the learnt function to tune the learning_rate and epoches
    if debug:
        x_val = np.linspace(0, 1, 100)
        fig = plt.figure(figsize = (8, 8))
        ax1 = fig.add_subplot(111)
        ax1.scatter(bias_x, bias_y, s = 0.2)
        for i in range(model.order):
            if i == 0:
                y_val = theta_bar[i] * (x_val ** i)
            else:
                y_val += theta_bar[i] * (x_val ** i)
        ax1.plot(x_val, y_val, 'r')
        plt.savefig("debug_plot/" + str(order) + ".png")
        
    return E_in_bar, E_out_bar, E_bias

def debug():
    """
    for debug:
        to tune the parameters in order to check whether the model could learn or not.
    """
    N = 50
    sigma = 0.1
    learning_rate = 0.1
    regularization = False
    debug = True

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    d_all_s = [3, 10, 21]
    for i in tqdm(range(len(d_all_s))):
        d = d_all_s[i]
        E_in_bar, E_out_bar, E_bias = experiment(sess, d, N, sigma, learning_rate, debug = debug, regularization = regularization)

def main(current_test, regularization = False):
    """
    Main Function:
        given the changing variables and find out results.

    Why write a main() function:
        for using multi-processing to speed up the training.
    """
    # requirements as the problem descriptions
    N_all = np.array([2, 5, 10, 20, 50, 100, 200])
    d_all = np.arange(21)
    sigma_all = np.array([0.01, 0.1, 1])

    # set a switch case for switch the questions quickly
    all_changing_data = [N_all, d_all, sigma_all]
    switcher = {"test_N": 0,
                "test_d": 1,
                "test_sigma": 2}
    test_num = switcher[current_test]

    # prepare the array for ploting
    E_in_plot = []
    E_out_plot = []
    E_bias_plot = []

    total_len = len(all_changing_data[test_num])

    print(total_len)
    
    # specify the memory management of GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # start experiment()
    for ind in tqdm(range(total_len)):
        # set all parameters
        if test_num == 0:
            N = N_all[ind]
            d = 5
            sigma = 0.1
        elif test_num == 1:
            N = 50
            d = d_all[ind]
            sigma = 0.1
        else:
            N = 200
            d = 10
            sigma = sigma_all[ind]
        learning_rate = 0.1

        # run experiment()
        E_in_bar, E_out_bar, E_bias = experiment(sess, d, N, sigma, learning_rate, ind, regularization = regularization)

        # store the values
        E_in_plot.append(E_in_bar)
        E_out_plot.append(E_out_bar)
        E_bias_plot.append(E_bias)

    # change into numpy array for easy save and read
    E_in_plot = np.array(E_in_plot)
    E_out_plot = np.array(E_out_plot)
    E_bias_plot = np.array(E_bias_plot)

    if test_num == 0:
        N = "all"
    elif test_num == 1:
        d = "all"
    else:
        sigma = "all"

    if regularization:
        reg = "_regularized"
    else:
        reg = "_noreg"

    np.save("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + reg + "_Ein.npy", E_in_plot)
    np.save("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + reg + "_Eout.npy", E_out_plot)
    np.save("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + reg + "_Ebias.npy", E_bias_plot)

if __name__ == "__main__":
    # multi-processing
    """
    3 modes, for 3 questions respectively:
        1. "test_N"
        2. "test_d"
        3. "test_sigma"
    """
    current_test = "test_d"
    regularization = False

    p = Process(target = main, args=(current_test, regularization))
    p.start()
    p.join()

    # for debugging only.
    # debug()

