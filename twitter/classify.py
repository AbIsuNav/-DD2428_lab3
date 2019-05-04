import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from twiset import TwitterDataset
from logreg import LogisticRegression


TRAIN_FILES = glob.glob(os.path.join('TwitterData', 'train', '*.json'))
TEST_FILES = glob.glob(os.path.join('TwitterData', 'test', '*.json'))
ds = TwitterDataset(TRAIN_FILES,True)
tds = TwitterDataset(TEST_FILES)


# ======= Run Multinomial Logistic Regression =======
b = LogisticRegression()
#b.fit(ds)
b.minibatch_fit(ds)
b.classify_datapoints(tds)
plt.show(block=True)
# ===================================================