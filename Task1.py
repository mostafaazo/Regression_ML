#%% Define Library
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import string
import random

#%% Generate Random Dataset

n_sample = 1000

# Generate Random Numeric Columns
_X, _y= datasets.make_regression(n_samples=n_sample,#number of samples
                                      n_features=2,#number of features
                                      noise=10,#bias and standard deviation of the guassian noise
                                      random_state=0) #set for same data points for each run
# Generate Random Letter Column
alphabet = list(string.ascii_lowercase)
letter_col = random.choices(alphabet,k=n_sample)

# Data Set
df = pd.concat([pd.DataFrame(_X,columns=["est_num1","est_num2"])
                   ,pd.DataFrame(letter_col,columns=["est_cat"])
                   ,pd.DataFrame(_y,columns=["target"])]
               , axis=1)


