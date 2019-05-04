import random
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
from twiset import TwitterDataset


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Dmytro Kalpakchi.
"""
class LogisticRegression(object):
    """
    This class performs logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    def __init__(self, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param theta    A ready-made model
        """
        theta_check = theta is not None

        if theta_check:
            self.FEATURES = len(theta)
            self.theta = theta
        #For Minibatch True : LR = 0.1, Reg = 0.001
        #  ------------------------ Hyperparameters ------------------------ #
        self.LEARNING_RATE = 0.1            # The learning rate.
        self.CONVERGENCE_MARGIN = 1e-8      # The convergence criterion.
        self.MAX_ITERATIONS = 10000         # Maximal number of passes through the datapoints
                                            # in minibatch gradient descent.
        self.MINIBATCH_SIZE = 100           # Minibatch size (only for minibatch gradient descent)
        self.EMB_SIZE = 50                  # Word embedding size (if applicable)
        self.MODE = 'bag-of-words'          # How to transform words into numeric vectors
        self.PATIENCE = 5                   # A max number of consequent epochs with monotonously
                                            # increasing validation loss for declaring overfitting
        # ------------------------------------------------------------------ #
        self.vocab = set()

        self.cat2id = {}

        self.lambda_reg = 0.001

        self.id2cat = {}


    def ds2xy(self, ds, mode='bag-of-words',flag=False):
        """
        Transforms a dataset to a set of variables X and Y. 
        X contains all datapoints, i.e. its dimensionality is (N, D + 1),
            where N is the number of datapoints, D is the number of features
            one extra column is a column full of 1s (corresponding to the bias term)
        Y contains labels for all datapoints, i.e. it's an array of length N

        @param ds   An instance of TwitterDataset class to be converted
        """
        if type(ds) != TwitterDataset:
            raise NotImplementedError("Unsupported type of a dataset")
        if mode == 'bag-of-words':
            N = ds.no_of_dp
            if flag:
                D = ds.no_of_unique_words + 1
                self.vocab = list(ds.vocab)
                self.cat2id = ds.cat2id
                self.id2cat = ds.id2cat
            else:
                D = self.FEATURES

            stoch = random.sample(range(N), N)  # random take datapoints

            x, y = np.zeros((N, D)), np.zeros(N, dtype=int)
            for i in range(N):
                x[i][0] = 1 # bias
                y[i] = ds.inverted_index.get(stoch[i])  # save class of data i
                txt = ds.data[stoch[i]]  # words in txt
                for j in range(len(txt)):  # check words in txt
                    try:
                        idx = self.vocab.index(txt[j])
                        x[i][idx+1] += 1
                    except:
                        pass
                #x[i][1:] /= len(txt)
            #pass
        else:
            raise NotImplementedError("Mode {} is not supported".format(mode))
        #print(y)
        return x, y


    def train_validation_split(self, x, y, ratio=0.9):
        """
        Splits the data into training and validation set, taking the `ratio` * 100 percent of the data for training
        and `1 - ratio` * 100 percent of the data for validation.

        @param x        A (N, D + 1) matrix containing training datapoints
        @param y        An array of length N containing labels for the datapoints
        @param ratio    Specifies how much of the given data should be used for training

        @returns        A 4-tuple containing (x_training, y_training, x_validation, y_validation)
        """
        #
        # YOUR CODE HERE
        #
        spl = int(len(x)*0.9) #datapints
        xt = x[:spl,:]
        yt = y[:spl]
        xv = x[spl:,:]
        yv = y[spl:]
        return xt, yt, xv, yv


    def init_params(self, ds):
        """
        Initializes the trainable parameters of the model and dataset-specific variables
        """
        # To limit the effects of randomness
        np.random.seed(524287)

        # x - Encoding of the data points (as a DATAPOINTS x FEATURES size array).
        # y - # Correct labels for the datapoints.
        x, y = self.ds2xy(ds, mode=self.MODE,flag=True)

        # Number of features
        self.FEATURES = len(x[0])

        # Number of classes
        self.CLASSES = len(set(y))

        # Training data is stored in self.x and self.y        
        self.x, self.y, self.xv, self.yv = self.train_validation_split(x, y)

        # Number of datapoints.
        self.DATAPOINTS = len(self.x)

        # The weights we want to learn in the training phase.
        self.theta = np.random.uniform(-0.5, 0.5, (self.FEATURES, self.CLASSES))

        # The current gradient.
        self.gradient = np.zeros((self.FEATURES, self.CLASSES))



    def loss(self, x, y):
        """
        Calculates the loss for the datapoints present in `x` given the labels `y`.
        datapoint size NxD+1
        y = size N
        :return loss
        """
        N = len(x)
        loss = 0.0
        for i in range(N):
            loss += -np.log(self.conditional_prob(y[i], x[i]))
        return loss/N


    def conditional_prob_m(self, label, datapoint):
        """
        datapoint size NxD+1
        label size 1
        Computes the conditional probability P(label|datapoint)
        softmax()
        :return 1xN
        """
        sum = np.zeros(len(datapoint))

        for i in range(self.CLASSES):
            sum += np.exp(np.matmul(np.transpose(self.theta[:,i]),np.transpose(datapoint)))

        s = np.exp(np.matmul(np.transpose(self.theta[:,label]),np.transpose(datapoint)))/sum

        return s

    def conditional_prob(self, label, datapoint):
        """
        datapoint size 1xD+1
        label size 1
        Computes the conditional probability P(label|datapoint)
        softmax()
        :return 1xN
        """
        sum = 0.0

        for i in range(self.CLASSES):
            sum += np.exp(np.matmul(np.transpose(self.theta[:,i]),datapoint[:]))

        return np.exp(np.matmul(np.transpose(self.theta[:,label]),datapoint[:]))/sum


    def compute_gradient_for_all(self,y):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        :return
        """
        s = np.dot(np.transpose(self.theta), np.transpose(self.x))
        # apply softmax activation function
        p = self.softmax(s)
        # predicted class is label with highest probability
        # k = np.argmax(p, axis=0)
        self.gradient = np.transpose(np.matmul(-((y) - p), self.x) / self.DATAPOINTS)
        pass




    def compute_gradient_minibatch(self, minibatch,y):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """
        # calculate the output
        s = np.dot(np.transpose(self.theta), np.transpose(minibatch))
        # apply softmax activation function
        p = self.softmax(s)
        # predicted class is label with highest probability
        #k = np.argmax(p, axis=0)
        self.gradient = np.transpose(np.matmul(-(np.transpose(y)-p),minibatch)/self.MINIBATCH_SIZE)

        pass

    def softmax(self, out):
        """
        Softmax activation function
        :return probabilities of the sample being in each class
        """
        e_out = np.exp(out - np.max(out))
        return e_out / e_out.sum(axis=0)

    def minibatch_fit(self, train_ds):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_params(train_ds)

        self.init_plot(self.FEATURES)

        it = 0
        cont = 0
        batch_epochs = int(self.DATAPOINTS / self.MINIBATCH_SIZE)
        prev_loss = 0
        y_onehot = np.zeros((self.CLASSES, self.DATAPOINTS))
        for i in range(self.DATAPOINTS):
            y_onehot[self.y[i]][i]=1
        while True:
            # random shuffle to the data
            permutation = np.random.permutation(self.DATAPOINTS)
            self.x = self.x[permutation, :]
            self.y = self.y[permutation]
            y_onehot = y_onehot[:,permutation]
            # train minibatches
            for batch in range(batch_epochs):
                start = batch * self.MINIBATCH_SIZE
                end = start + self.MINIBATCH_SIZE
                batch_data = self.x[start:end]
                tempy = np.transpose(y_onehot)[start:end]
                self.compute_gradient_minibatch(batch_data,tempy)
                self.theta = self.theta - self.LEARNING_RATE * (self.gradient + 2 * self.lambda_reg * self.theta)

            # Calculate and display training and validation losses
            tr_loss, val_loss = self.loss(self.x, self.y), self.loss(self.xv, self.yv)
            # early stop
            if val_loss > prev_loss:
                if cont >= self.PATIENCE:
                    break
                cont += 1
            else:
                cont = 0
            prev_loss = val_loss
            # down scale learning rate
            if it%200 == 0:
                self.LEARNING_RATE *= 0.9
                print("LEARNING_RATE: ",self.LEARNING_RATE)
            # update plot
            if it % 10 == 0:
                self.update_plot(*[tr_loss, val_loss])

            # STOP when reached max iterations
            if it > self.MAX_ITERATIONS:
                break
            it += 1


    def fit(self, train_ds):
        """
        Performs Batch Gradient Descent
        """
        self.init_params(train_ds)

        self.init_plot(self.FEATURES)
        it = 0
        print("FEATURES: ",self.FEATURES)
        cont = 0
        prev_loss = 0
        y_onehot = np.zeros((self.CLASSES, self.DATAPOINTS))
        for i in range(self.DATAPOINTS):
            y_onehot[self.y[i]][i] = 1
        while True:
            #shuffle random
            permutation = np.random.permutation(self.DATAPOINTS)
            self.x = self.x[permutation,:]
            self.y = self.y[permutation]
            y_onehot = y_onehot[:, permutation]

            self.compute_gradient_for_all(y_onehot)
            self.theta = self.theta - self.LEARNING_RATE * (self.gradient + 2 * self.lambda_reg * self.theta)
            tr_loss, val_loss = self.loss(self.x, self.y), self.loss(self.xv, self.yv)

            # early stop
            if val_loss > prev_loss:
                if cont >= self.PATIENCE:
                    break
                cont += 1
            else:
                cont = 0
            prev_loss = val_loss
            if it%100 == 0:
                self.LEARNING_RATE *= 0.9
                print("LEARNING_RATE: ",self.LEARNING_RATE)
            self.update_plot(*[tr_loss, val_loss])

            # REPLACE THE CODE BELOW
            if it > 1000:#self.MAX_ITERATIONS:
                break
            it += 1
        #print("Theta ", self.theta)



    def classify_datapoints(self, test_ds):
        """
        Classifies datapoints
        """
        x, y = self.ds2xy(test_ds, mode=self.MODE)
        confusion = np.zeros((self.CLASSES, self.CLASSES))
        prob = np.transpose(self.softmax(np.dot(np.transpose(self.theta), np.transpose(x)))) # NxK
        for d in range(test_ds.no_of_dp):
            best_class = np.argmax(prob[d])
            confusion[best_class][y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(self.CLASSES)))
        for i in range(self.CLASSES):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(self.CLASSES)))
        if test_ds.no_of_dp > 0:
            acc = sum([confusion[i][i] for i in range(self.CLASSES)]) / test_ds.no_of_dp
        else:
            acc = 0
        print("Accuracy: {0:.2f}%".format(acc * 100))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))

    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)