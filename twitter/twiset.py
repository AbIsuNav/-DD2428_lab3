import re
import json
import string
from collections import defaultdict


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Dmytro Kalpakchi.
"""
class Datapoint(object):
    """
    This class represents a text as a bag-of-words + a category.
    """
    def __init__(self, tokens, label):
        # The category.
        self.label = label

        self.tokens = tokens

        # The text represented as a map from the word to the number of occurrences.
        self.tcount = defaultdict(int)

        self.no_of_words = len(tokens)

        for token in tokens:
            self.tcount[token] += 1


class TwitterDataset(object):
    """
    This class represents a set of datapoints (= texts + categories).
    """
    def __init__(self, filenames):
        # The number of word occurrences per category.
        self.no_of_words_per_cat = defaultdict(int)

        # The number of data points per category.
        self.no_of_dp_per_cat = defaultdict(int)

        # The number of categories (=classes).
        self.no_of_cat = 0

        # Datapoints
        self.data = []

        # Vocabulary
        self.vocab = set()

        # Categories
        self.cat_names = list()

        # A mapping from categories to their IDs
        self.cat2id = {}

        # Read data
        
        #########################################
        # REPLACE THE CODE BELOW WITH YOUR CODE #
        #########################################
        self.cat_names = ['Category 1', 'Category 2', 'Category 3']
        self.cat2id = {c: i for i, c in enumerate(self.cat_names)}
        self.no_of_cat = len(self.cat_names)

        #########################################

        # Number of datapoints
        self.no_of_dp = len(self.data)

        # Number of unique features
        self.no_of_unique_words = len(self.vocab)

