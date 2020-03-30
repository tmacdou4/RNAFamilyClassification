import pandas as pd
import numpy as np
from test_loader import *
import pdb
import os

# global variables
# paths
datapath = 'data'
# stats
seq_len = 500 # how to get this number efficiently ?
# Set RFs to include 
RFs = ['RF00005', 'RF00009']
# loading data into data frame
data, labels = load_data_in_df(RFs, datapath = datapath, seq_len = seq_len)
# debug
pdb.set_trace()
