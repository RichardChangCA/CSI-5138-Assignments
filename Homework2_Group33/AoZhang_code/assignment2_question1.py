"""
CSI 5138: Assignment 2 ----- Question 1
Student:            Ao   Zhang
Student Number:     0300039680
"""
#### for plotting through X11 #####
import matplotlib
matplotlib.use("tkagg")
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#### other dependencies #####
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from multiprocessing import Process


class QuestionOne:
    """
    Using tensorflow building the model
    """
    def __init__(self, N, K):
        """
        Args:
            k               ->              dimension of input vector X
            N               ->              batch size of the input
        """ 
        self.K = K
        self.batch_size = N
        # build the place holder for the input
        self.X = tf.placeholder(tf.float32, shape = (None, self.K, 1))
        # define variables
        self.A = tf.Variable(tf.glorot_uniform_initializer()((self.K, self.K)))
        self.B = tf.Variable(tf.glorot_uniform_initializer()((self.K, self.K)))
        # repeat the weights into the shape (batch_size, weights_shape) for further computation
        self.A_batch = self.WeightsBatch(self.A)
        self.B_batch = self.WeightsBatch(self.B)
        # define a learning rate
        self.learning_rate = 0.001

    def WeightsBatch(self, weights):
        """
        Function: copy the weights from shape (K, K) to shape (batch_size, K, K) 
        """
        return tf.ones([self.batch_size, 1., 1.]) * weights

    ###################################################################
    # define all functions and its relative gradients
    ###################################################################
    def FuncLinear(self, var1, var2):
        """
        Function: y = A * x
        """
        return tf.matmul(var1, var2)

    def GradientLinear(self, var1, var2):
        """
        Function: \partial{grad(y)}{var1} = x; \partial{grad(y)}{var1} = A; 
        """
        return tf.transpose(var2, perm = [0, 2, 1]), \
                tf.transpose(var1, perm = [0, 2, 1])

    def Sigmoid(self, var):
        """
        Function: Sigmoid
        """
        return 1. / (1. + tf.exp( - var))

    def GradientSigmoid(self, var):
        """
        Function: grad(sigmoid) = sigmoid * (1 - sigmoid)
        """
        return self.Sigmoid(var) * (1 - self.Sigmoid(var))

    def FuncMultiplication(self, var1, var2, var3):
        """
        Function: y = A * (u * v)
        """
        return tf.matmul(var1, (var2 * var3))

    def GradientMultiplication(self, var1, var2, var3):
        """
        Function: \partial{grad(y)}{var1} = var2 * var3
        """
        return tf.transpose(var2 * var3, perm = [0, 2, 1]), \
                tf.matmul(var1, var3), tf.matmul(var1, var2)

    def EuclideanNorm(self, var):
        """
        Function: Euclidean Norm(X)
        """
        return tf.reduce_sum(tf.square(var))
    
    def GradientEuclideanNorm(self, var):
        """
        Function: 2*x_1, 2*x_2, ... , 2*x_n
        """
        return 2 * var

    ###################################################################
    ## calculate the forward graph, gradient graph and dual gradient ##
    ###################################################################
    def ForwardGradientGraph(self, name = "gradient"):
        """
        Function: Calculate loss function and forward gradient
        """
        # forward loss function computation
        y = self.FuncLinear(self.A_batch, self.X)
        u = self.Sigmoid(y)
        v = self.FuncLinear(self.B_batch, self.X)
        z = self.FuncMultiplication(self.A_batch, u, v)
        omega = self.FuncLinear(self.A_batch, z)
        loss = self.EuclideanNorm(omega)

        # forward gradient computation for each edge
        grad_y_A, grad_y_X = self.GradientLinear(self.A_batch, self.X)
        grad_u_y = self.GradientSigmoid(y)
        grad_v_B, grad_v_X = self.GradientLinear(self.B_batch, self.X)
        grad_z_A, grad_z_u, grad_z_v = self.GradientMultiplication(self.A_batch, u, v)
        grad_omega_A, grad_omega_z = self.GradientLinear(self.A_batch, z)
        grad_loss_omega = self.GradientEuclideanNorm(omega)

        # transpose of forward gradient graph (back propogation) w.r.t parameter A
        grad_A = tf.matmul(tf.matmul(grad_omega_z, grad_loss_omega) * grad_z_u * grad_u_y, grad_y_A) \
                + tf.matmul(tf.matmul(grad_omega_z, grad_loss_omega), grad_z_A) \
                + tf.matmul(grad_loss_omega, grad_omega_A)
    
        # transpose of forward gradient graph (back propogation) w.r.t parameter B
        grad_B = tf.matmul(tf.matmul(grad_omega_z, grad_loss_omega) * grad_z_v, grad_v_B)

        # define returns
        if name == "loss":
            return loss
        elif name == "gradient":
            return grad_A, grad_B
        else:
            raise ValueError("Namescope is wrong, please doublecheck the arguments")

    def BackPropGradientDescent(self):
        """
        Function: Apply GD based on back propagation
        """
        # get the gradient of A and B
        grad_L_A, grad_L_B = self.ForwardGradientGraph(name = "gradient")

        # gradient descent
        learning_A = tf.add(self.A, - tf.reduce_mean(self.learning_rate * grad_L_A, \
                                                    axis = 0))
        learning_B = tf.add(self.B, -  tf.reduce_mean(self.learning_rate * grad_L_B, \
                                                    axis = 0))
        
        # assign values
        operation_A = self.A.assign(learning_A)
        operation_B = self.B.assign(learning_B)
        return operation_A, operation_B


###################################################################
####### random test, feel free to tune all those parameters #######
###################################################################
if __name__ == "__main__":

    # input size
    N = 100
    # input dimension
    K = 5

    # use matplotlib instead of TensorBoard to plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # start building model
    Q_one =  QuestionOne(N, K)

    # produce random data
    X_data = np.random.randint(10, size = (N, K, 1))

    # tensorflow settings (GPU settings)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initializers.global_variables()
    sess.run(init)

    # get loss value for visualize
    loss = Q_one.ForwardGradientGraph(name = "loss")
    # updating weights
    opA, opB = Q_one.BackPropGradientDescent()

    indall = []
    lossall = []
    for i in range(100):
        _, _, loss_val = sess.run([opA, opB, loss], feed_dict = {Q_one.X: X_data})
        print(loss_val)

        indall.append(i)
        lossall.append(loss_val)


    ax1.plot(indall, lossall)
    ax1.grid()
    ax1.set_title("value of loss function in 100 training steps")
    ax1.set_xlabel("training steps")
    ax1.set_ylabel("loss value")
    plt.savefig("assignment1_question1.png")
    plt.show()     


    



