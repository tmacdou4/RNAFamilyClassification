import time
import pandas as pd
import numpy as np
import warnings
import argparse
from utility import *
from models import *
import pdb
import os
import torch
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader 

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# parse arguments 
parser.add_argument('-arch', dest = 'ARCH', default = [100, 200], nargs = 2, type = int, help = 'set the architecture of the 1 or 2 - layers DNN model. ex 100 in 1st , 2000 in second is typed : -arch 100 2000')
parser.add_argument('-epochs',dest = 'EPOCHS', default = 3, type = int, help = 'nb of max epochs')
parser.add_argument('-wd', dest = 'WEIGHT_DECAY', type = float, default = 0.2, help = 'L2 parametrization [0:no regularization]')
parser.add_argument('-xval', dest = 'XVAL', default = 5, type = int, help= 'number of folds for crossvalidation')
parser.add_argument('-t', dest = 'TARGET', default = 'RF01852', type = str, help = 'name of label to train on') 
parser.add_argument('-seed', dest = 'SEED', default= 1, type = int, help = 'random seed')
args = parser.parse_args()


# some other global variables
# paths
datapath = 'data'
modelname = 'DNN'
ARCH = args.ARCH
# static values 
vs = 'REST'
bs = 128
seq_len = 600 # how to get the optimal number efficiently ?
# Set RFs to include
RFs =[path for path in os.listdir(datapath) if os.path.isdir(os.path.join(datapath,path))] 

# loading data into data frame
data, labels = load_data_in_df(RFs, datapath = datapath, max_len = seq_len)
np.random.seed(args.SEED) # set numpy seed for data shuffle  
# multiclass ONLY
# numeric_labels = dict(zip(np.unique(labels['RFAM']), np.arange(len(np.unique(labels['RFAM'])))))
# labels['numeral'] = [numeric_labels[l] for l in labels['RFAM']]
labels['numeral'] = np.array(labels['RFAM'] == args.TARGET, dtype = int)
torch.manual_seed(args.SEED) # set torch seed for model initialization 
rnd_idxs = np.arange(labels.shape[0]) # get ids 
np.random.shuffle(rnd_idxs)    # shuffles ids
labels = labels.iloc[rnd_idxs] # shuffle labels
data = data.iloc[rnd_idxs] # shuffle data

nseeds = labels.shape[0]
test_size = int(float(labels.shape[0]) / args.XVAL)
train_size = nseeds - test_size
gr_steps = int(float(train_size) / bs) + 1
# define model specs
model_specs = {
                'xval' : args.XVAL, 
                'nseeds' : nseeds,
                'test_size': test_size,                  
                'train_size' : train_size,
                'gr_steps': gr_steps,
                'model_layout': 'DNN', 
                'n_hid_lyrs': len(ARCH),
                'loader': 'balanced',
                'target': args.TARGET,
                'vs' : vs,
                'seq_len': seq_len,
                'input_size': data.shape[1],
                'batch_size' : bs, # train_size / 10 ,
                'wd' : args.WEIGHT_DECAY, 
                'lr': 1e-4,
                'drp-1': 0,
                'HID1N': ARCH[0],
                'HID2N': ARCH[1],
                'ARCHID': '.'.join([str(e) for e in ARCH]),
                'MODID': 'DEBUG',
                'RFID': '{}_{}'.format(args.TARGET, vs),
                'lossfn': torch.nn.BCELoss(),
                'epochs': args.EPOCHS,
                'levels' : max(labels['numeral']) + 1,
                'output_size' : max(labels['numeral']) + 1,
                'device' : 'cuda:0',
                'training_acc' : None
                }
# OUTDATED init file tree for reporters
#target_path = os.path.join(modelname, args.TARGET)
#progress_path = os.path.join(target_path, 'PROGRESS')
#models_path = os.path.join(target_path, 'MODELS')
#trainlog_path = os.path.join(target_path, 'TRAINING_LOGS')
#res_path = os.path.join(target_path, 'RES')
#
#assert_mkdir(target_path)
#assert_mkdir(progress_path)
#assert_mkdir(models_path)
#assert_mkdir(trainlog_path)
#assert_mkdir(res_path)

# prepare_outfile_paths

SETS_path = os.path.join(model_specs['model_layout'], 'SETS')
MODELSPECS_path = os.path.join(model_specs['model_layout'], 'MODELSPECS')
MODELS_path = os.path.join(model_specs['model_layout'], 'MODELS')
assert_mkdir(SETS_path)
assert_mkdir(MODELSPECS_path)
assert_mkdir(MODELS_path)

# define train function 
from train import train

# foreach fold in xval
for foldn in range(1 , args.XVAL + 1):
    print('fold {} / {}'.format(foldn, args.XVAL)) 
    # store some static values 
    nsamples = model_specs['nseeds']
    test_size = model_specs['test_size']
    MODELFULLNAME = '_'.join(np.array([model_specs['RFID'], model_specs['ARCHID'], model_specs['MODID'], foldn],dtype = str ))
    # split train and test
    # prepare data splitting    
    samplesID = range(foldn * test_size , min((foldn + 1) * test_size, nsamples))
    
    TEST_X = data.iloc[samplesID]
    TEST_Y = labels.iloc[samplesID]
    # write TESTSET to disk
    print('writing test set seed {} fold {} / {} ...'.format(args.SEED, foldn, args.XVAL))
    TEST_SET = data.iloc[samplesID].join(labels.iloc[samplesID])
    TEST_SET.to_csv(os.path.join(SETS_path, 'SEED{}_F{}_{}.csv'.format(args.SEED, foldn, args.XVAL)))
    TRAIN_X = data.iloc[np.setdiff1d(labels.index, samplesID)]
    TRAIN_Y = labels.iloc[np.setdiff1d(labels.index, samplesID)]
    # init dataset objects 
    # dataset = Dataset({'data': np.array(TRAIN_X),'labels':np.array(TRAIN_Y.numeral)})
    dataset  = BalancedDataPicker({'data': np.array(TRAIN_X),'labels':np.array(TRAIN_Y.numeral)[np.newaxis].T }) 
    dl = DataLoader(dataset, batch_size = model_specs['batch_size']) 
    # init model
    model = DNN(model_specs).to(model_specs['device'])
    # train model
    print('training model...')
    # time stamp
    startime = time.clock()
    train(model, dl,  model_specs, device = model_specs['device'], foldn=foldn)
    # model is trained, record time, prepare to save model under a REFID
    # save torch model on disk under the name MODELFULLNAME.txt
    torch.save(model.state_dict(), os.path.join(MODELS_path, '{}.txt'.format(MODELFULLNAME)))
    # save up some reported values
    updates = [('train_proc_time', time.clock() - startime)
              ]
    # update model_specs with various reports
    model_specs.update(updates)
    # save model_specs dict under the name MODELFULLNAME.specs 
    with open(os.path.join(MODELSPECS_path, '{}.specs'.format(MODELFULLNAME)), 'w') as o : o.write(str(model_specs)) # to be updated 


