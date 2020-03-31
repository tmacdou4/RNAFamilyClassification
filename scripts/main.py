import pandas as pd
import numpy as np
from utility import *
import pdb
import os
import torch

# global variables
# paths
datapath = 'data'
# stats
seq_len = 200 # how to get this number efficiently ?
# Set RFs to include 
RFs = ['RF00005', 'RF01852'] # tRNA might not be an easy task 
# loading data into data frame
data, labels = load_data_in_df(RFs, datapath = datapath, seq_len = seq_len)
# debug
pdb.set_trace()

# define model specs

# define model architecture 

# foreach fold in xval

    # split train and test

    # init model

    # init reporter

    # train 
        # report at each epoch

    # test
        # report/plot





