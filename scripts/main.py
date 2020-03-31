import pandas as pd
import numpy as np
import warnings
import argparse
from utility import *
import pdb
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# parse arguments 
parser.add_argument('-arch', dest = 'ARCH', default = [50, 20], nargs = 2, type = int, help = 'set the architecture of the 1 or 2 - layers DNN model. ex 100 in 1st , 2000 in second is typed : -arch 100 2000')
parser.add_argument('-epochs',dest = 'EPOCHS', default = 50, type = int, help = 'nb of max epochs')
parser.add_argument('-wd', dest = 'WEIGHT_DECAY', type = float, default = 0.2, help = 'L2 parametrization [0:no regularization]')
parser.add_argument('-xval', dest = 'XVAL', default =10, type = int, help= 'number of folds for crossvalidation')
parser.add_argument('-t', dest = 'TARGET', default = 'RFAM', type = str, help = 'name of label to train on') 
parser.add_argument('-seed', dest = 'SEED', default= 1, type = int, help = 'random seed')
args = parser.parse_args()


# some other global variables
# paths
datapath = 'data'
modelname = 'DNN'


# stats
seq_len = 200 # how to get this number efficiently ?
# Set RFs to include 
RFs = ['RF00005', 'RF01852'] # tRNA might not be an easy task 
# loading data into data frame
data, labels = load_data_in_df(RFs, datapath = datapath, seq_len = seq_len)

# define model specs
model_specs = {
                'xval' : args.XVAL, 
                'nseeds' : labels.shape[0],
                'test_size': int(float(labels.shape[0]) / args.XVAL),
                'modelname': modelname,
                'n_hid_lyrs': len(args.ARCH),
                'loader': 'balanced',
                'target': args.TARGET,
                'input_size': data.shape[0],
                'output_size' : len(np.unique(labels[args.TARGET])),
                'batch_size' : 64, # train_size / 10 ,
                'wd' : args.WEIGHT_DECAY, 
                'lr': 1e-4,
                'drp-1': 0,
                'HID1N': args.ARCH[0],
                'HID2N': args.ARCH[1],
                'lossfn': torch.nn.CrossEntropyLoss(),
                'epochs': args.EPOCHS,
                'levels' : np.unique(labels[args.TARGET])
                }

# define model architecture 
class DNN(nn.Module):
        def __init__(self,model_specs):
                super(DNN, self).__init__()
                self.in_h1 = nn.Linear(model_specs['input_size'], model_specs['HID1N'])
                self.h1_nl = nn.Hardtanh()
                self.h1_h2 = nn.Linear(model_specs['HID1N'], model_specs['HID2N'])
                self.h2_nl = nn.Hardtanh()
                self.h2_out = nn.Linear(model_specs['HID2N'], model_specs['output_size'])
                self.out_nl = nn.Softmax(dim=-1)
        def forward(self, x):
            out = self.in_h1(x)
            out = self.h1_nl(out)
            out = self.h1_h2(out)
            out = self.h2_nl(out)
            out = self.h2_out(out)
            out = self.out_nl(out)
            return out 

# init file tree for reporters
# prepare_outfile_paths
target_path = os.path.join(modelname, args.TARGET)
progress_path = os.path.join(target_path, 'PROGRESS')
models_path = os.path.join(target_path, 'MODELS')
trainlog_path = os.path.join(target_path, 'TRAINING_LOGS')
res_path = os.path.join(target_path, 'RES')

assert_mkdir(target_path)
assert_mkdir(progress_path)
assert_mkdir(models_path)
assert_mkdir(trainlog_path)
assert_mkdir(res_path)


# foreach fold in xval
for foldn in range(args.XVAL):
    # make training log outpath
    TRAINING_outpath = os.path.join(trainlog_path, str(foldn).zfill(3))
    assert_mkdir(TRAINING_outpath)
    # store some static values 
    nsamples = model_specs['nseeds']
    test_size = model_specs['test_size']
    # split train and test
    # prepare data splitting    
    samples = np.array(labels.index[foldn * test_size: min((foldn + 1) * test_size, nsamples)], dtype = str)
    TEST_X = data.loc[samples]
    TEST_Y = labels.loc[samples]
    TRAIN_X = data.loc[set(labels.index) - set(samples)]
    TRAIN_Y = labels.loc[set(labels.index) - set(samples)]
    
    # init reporter

    # train 
        # report at each epoch

    # test
        # report/plot
