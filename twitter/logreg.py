import random
import numpy as np
import matplotlib.pyplot as plt

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


    def ds2xy(self, ds, mode='bag-of-words'):
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

        x, y = [[0]*10]*10, [0]*3 + [1]*3 + [2]*4
        if mode == 'bag-of-words':
            #
            # YOUR CODE HERE
            #
            pass
        else:
            raise NotImplementedError("Mode {} is not supported".format(mode))
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
        return x, y, x, y


    def init_params(self, ds):
        """
        Initializes the trainable parameters of the model and dataset-specific variables
        """
        # To limit the effects of randomness
        np.random.seed(524287)

        # x - Encoding of the data points (as a DATAPOINTS x FEATURES size array).
        # y - # Correct labels for the datapoints.
        x, y = self.ds2xy(ds, mode=self.MODE)

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
        """
        #
        # YOUR CODE HERE
        #
        return 0


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        #
        # YOUR CODE HERE
        #
        return 0.5


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        #
        # YOUR CODE HERE
        #
        pass


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """
        #
        # YOUR CODE HERE
        #
        pass


    def minibatch_fit(self, train_ds):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_params(train_ds)

        self.init_plot(self.FEATURES)

        it = 0
        #
        # YOUR CODE HERE
        #
        while True:
            #
            # YOUR CODE HERE
            #
            
            # Calculate and display training and validation losses
            tr_loss, val_loss = self.loss(self.x, self.y), self.loss(self.xv, self.yv)
            if it % 10 == 0:
                self.update_plot(*[tr_loss, val_loss])
            # REPLACE THE CODE BELOW
            if random.random() > 0.99:
                break
            it += 1


    def fit(self, train_ds):
        """
        Performs Batch Gradient Descent
        """
        self.init_params(train_ds)

        self.init_plot(self.FEATURES)

        it = 0
        #
        # YOUR CODE HERE
        #
        while True:
            #
            # YOUR CODE HERE
            #
            
            # Calculate and display training and validation losses
            tr_loss, val_loss = self.loss(self.x, self.y), self.loss(self.xv, self.yv)
            self.update_plot(*[tr_loss, val_loss])

            # REPLACE THE CODE BELOW
            if random.random() > 0.99:
                break
            it += 1


    def classify_datapoints(self, test_ds):
        """
        Classifies datapoints
        """
        x, y = self.ds2xy(test_ds, mode=self.MODE)

        confusion = np.zeros((self.CLASSES, self.CLASSES))

        for d in range(test_ds.no_of_dp):
            best_prob, best_class = -float('inf'), None
            dp = test_ds.data[d]
            for c in range(self.CLASSES):
                prob = self.conditional_prob(c, x[d])
                if prob > best_prob:
                    best_prob = prob
                    best_class = c
            confusion[best_class][self.cat2id[dp.label]] += 1

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