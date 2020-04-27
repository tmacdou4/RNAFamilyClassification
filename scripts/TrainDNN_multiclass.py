import matplotlib as mpl
mpl.use('Agg')
import  matplotlib.pyplot as plt
import time
import itertools as it
import pandas as pd
import numpy as np
import warnings
import argparse
from utility import *
from models_multiclass import *
import pdb
import os
import torch
from sklearn import metrics
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader
from train_multiclass import train
from torch.autograd import Variable
import seaborn as sn

#
# TrainDNNs.py
#

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# parse arguments 
parser.add_argument('-arch', dest = 'ARCH', default = [1000, 1000], nargs = '+', type = int, help = 'set the architecture of the n-layers DNN model. ex 100 in 1st , 2000 in second is typed : -arch 100 2000')
parser.add_argument('-epochs',dest = 'EPOCHS', default = 100 , type = int, help = 'nb of max epochs')
parser.add_argument('-wd', dest = 'WEIGHT_DECAY', type = float, default = 0, help = 'L2 parametrization [0:no regularization]')
parser.add_argument('-xval', dest = 'XVAL', default = 5, type = int, help= 'number of folds for crossvalidation')
parser.add_argument('-seed', dest = 'SEED', default= 1, type = int, help = 'random seed')
parser.add_argument('-d', dest = 'DEVICE', default= 'cuda:0', type = str, help = 'device ex cuda:0')
parser.add_argument('-task', dest = 'TASK', default= 'ZP', type = str, help ='type of dataset randomness / padding sequences [ZP, RP, NUCSHFLZP, NUCSHFLRP, DINUCSHFL]')
parser.add_argument('-target', dest = 'TARGET', default= 'RF00005', type = str, help = 'RFAM identifier to predict from seed')
parser.add_argument('-classification', dest = 'CLSFID', default='BIN', type = str, help ='BIN for binary classification, MUL for multiclass classification')

args = parser.parse_args()

# some other global variables
# paths
datapath = 'data'
# static values 
modelname = 'DNN'
lr = 1e-4
drp = 0
bs = 512
loader = 'balanced'
seq_len = 400 # how to get the optimal number efficiently ?

# Set RFs to include
RFs =[path for path in os.listdir(datapath) if os.path.isdir(os.path.join(datapath,path))]
RFs.sort()
# update some variables
modelID = 'WD{}_EP{}'.format(args.WEIGHT_DECAY, args.EPOCHS)
clsfID = args.CLSFID
taskID = args.TASK
target = args.TARGET
ARCH = args.ARCH
archID = '[' + ','.join([str(e) for e in ARCH]) + ']'

if clsfID == 'MUL':
    # multiclass data and labels
    data, labels = load_data_in_df(None, RFs, taskID, datapath=datapath, max_len=seq_len)
    numeric_labels = dict(zip(np.unique(labels['RFAM']), np.arange(len(np.unique(labels['RFAM'])))))
    labels['numeral'] = [numeric_labels[l] for l in labels['RFAM']]
    IDs = np.array([clsfID, taskID, archID, modelID], dtype=str)
else:
    # target vs rest data and labels
    data, labels = load_data_in_df(target, RFs, taskID, datapath=datapath, max_len=seq_len)
    labels['numeral'] = np.array(labels['RFAM'] == target, dtype=int)
    IDs = np.array([clsfID, target, taskID, archID, modelID], dtype=str)

MODELFULLNAME = '_'.join(IDs)

np.random.seed(args.SEED) # set numpy seed for data shuffle
torch.manual_seed(args.SEED) # set torch seed for model initialization 
rnd_idxs = np.arange(labels.shape[0]) # get ids 
np.random.shuffle(rnd_idxs)    # shuffles ids
labels = labels.iloc[rnd_idxs] # shuffle labels
data = data.iloc[rnd_idxs] # shuffle data
# static stats variables
nseeds = labels.shape[0]
test_size = int(float(nseeds) / args.XVAL)
train_size = nseeds - test_size
gr_steps = int(float(train_size) / bs) + 1

# prepare_outfile_paths
MODELSPECS_path = os.path.join(modelname, 'MODELSPECS')
MODELS_path = os.path.join(modelname, 'MODELS')
RES_path = os.path.join(modelname, 'OUT')
TRPLOTS_path = os.path.join(modelname, 'TRPLOTS')
assert_mkdir(MODELSPECS_path)
assert_mkdir(MODELS_path)
assert_mkdir(RES_path) 
assert_mkdir(TRPLOTS_path)

# init fig, axes for plotting 
fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (20, 11))

if clsfID == "BIN":
    total_conf_matrix = np.zeros((2, 2))
else:
    total_conf_matrix = np.zeros(shape=(len(np.unique(labels['RFAM'])),len(np.unique(labels['RFAM']))))

for foldn in range(1 , args.XVAL + 1):
    
    # define model specs
    model_specs = {
        'xval' : args.XVAL, 
        'shfl_seed' : args.SEED,
        'nseeds' : nseeds,
        'test_size': test_size,                  
        'train_size' : train_size,
        'gr_steps': gr_steps,
        'model_layout': modelname,
        'ARCH': ARCH, # needs update bcs it's array
        'n_hid_lyrs': len(ARCH),
        'n_free_params': None,
        'loader': loader,
        'seq_len': seq_len,
        'input_size': data.shape[1],
        'batch_size' : bs, # train_size / 10 ,
        'wd' : args.WEIGHT_DECAY, 
        'lr': lr,
        'drp': drp,
        'ARCHID': archID,
        'CLSFID': clsfID,
        'TASKID': taskID, 
        'MODID': modelID,
        'FOLDID': foldn, 
        'lossfn': torch.nn.NLLLoss(),
        'epochs': args.EPOCHS,
        'levels' : max(labels['numeral']) + 1,
        'output_size' : max(labels['numeral']) + 1,
        'device' : args.DEVICE,
        'tr_acc' : None,
        'tr_auc' : None,
        'tr_l' : None,
        'tr_proc_time': None,
        'tst_acc' : None, 
        'tst_auc' : None,
        'tst_l' : None,
        'conf_mat' : None
        }

    FOLD_NAME = MODELFULLNAME + "_" + str(foldn)

    if clsfID == 'MUL':
        print('({}) TASK: {} ARCH: {} MODEL: {} fold: {} / {}'.format(clsfID, taskID, ARCH, modelID, foldn, args.XVAL))
    else:
        print('RFID: {} ({}) TASK: {} ARCH: {} MODEL: {} fold: {} / {}'.format(target, clsfID, taskID, ARCH, modelID, foldn, args.XVAL))

    # split train and test
    # prepare data splitting
    nsamples = model_specs['nseeds']
    test_size = model_specs['test_size']
    samplesID = range((foldn - 1) * test_size , min((foldn) * test_size, nsamples))
    
    TEST_X = data.iloc[samplesID]
    TEST_Y = labels.iloc[samplesID]
    TRAIN_X = data.iloc[np.setdiff1d(labels.index, samplesID)]
    TRAIN_Y = labels.iloc[np.setdiff1d(labels.index, samplesID)]
    # init dataset objects 
    # dataset = Dataset({'data': np.array(TRAIN_X),'labels':np.array(TRAIN_Y.numeral)})
    tr_dataset = BalancedDataPicker({'data': np.array(TRAIN_X),'labels':np.array(TRAIN_Y.numeral)[np.newaxis].T })
    tr_dl = DataLoader(tr_dataset, batch_size = model_specs['batch_size'])

    val_dataset = ValidationDataPicker({'data': np.array(TEST_X),'labels':np.array(TEST_Y.numeral)[np.newaxis].T })
    val_dl = DataLoader(val_dataset, batch_size=model_specs['batch_size'])
    # init model
    model = DNN(model_specs).to(model_specs['device'])
    # store nb of params in model
    model_specs['n_free_params'] = get_n_params(model)
    # train model
    print('training model...')
    # time stamp
    startime = time.clock()
    train(model, tr_dl, val_dl, model_specs, device = model_specs['device'], foldn=foldn)
    # model is trained, record time, prepare to save model under a REFID
    # save torch model on disk under the name MODELFULLNAME.params
    torch.save(model.state_dict(), os.path.join(MODELS_path, '{}.params'.format(FOLD_NAME)))
    # save up some reported values
    # update model_specs with various reports
    model_specs['tr_proc_time'] = time.clock() - startime

    # store training curves and final accuracies on train
    nbsteps = len(model_specs['tr_l'])
    tr_losses = model_specs['tr_l']
    tr_accuracies = model_specs['tr_acc']
    tr_auc = model_specs['tr_auc']
    val_losses = model_specs['val_l']
    val_accuracies = model_specs['val_acc']
    val_auc = model_specs['val_auc']
    total_conf_matrix += model_specs['val_conf_mat']

    print("--------FOLD {}".format(foldn))
    acc_report = "FOLD NUMBER {} | FINAL TRAINING ACCURACY: {} | FINAL VALIDATION ACCURACY: {}".format(foldn, round(tr_accuracies[-1], 3), round(val_accuracies[-1], 3))
    loss_report = "FOLD NUMBER {} | FINAL TRAINING LOSS: {} | FINAL VALIDATION LOSS: {}".format(foldn, round(tr_losses[-1], 3), round(val_losses[-1], 3))
    auc_report = "FOLD NUMBER {} | FINAL TRAINING AUC: {} | FINAL VALIDATION LOSS: {}".format(foldn, round(tr_auc[-1], 3), round(val_auc[-1], 3))
    time_report =  "TRAINING/VALIDATION PROCESSING TIME:{}".format(model_specs['tr_proc_time'])
    print(acc_report)
    print(loss_report)
    print(auc_report)
    print(time_report)

    # Save training curves in OUT
    pd.DataFrame({'tr_loss': tr_losses, 'tr_acc': tr_accuracies, 'tr_auc': tr_auc}).to_csv(os.path.join(RES_path, FOLD_NAME + ".tr_curves"))
    pd.DataFrame({'val_loss': val_losses, 'val_acc': val_accuracies, 'val_auc': val_auc}).to_csv(os.path.join(RES_path, FOLD_NAME + ".val_curves"))

    # save model_specs dict under the name FOLD_NAME.specs in MODELSPECS
    with open(os.path.join(MODELSPECS_path, '{}.specs'.format(FOLD_NAME)), 'w') as o : o.write(str(model_specs)) # to be updated
    
    # PLOT RESULTS 
    # plot losses
    axes[0, 0].plot(np.arange(len(tr_losses)), tr_losses, lw = 1, label = "TRAIN, FOLD {}".format(foldn))
    axes[0, 0].plot(np.arange(len(val_losses)), val_losses, lw = 1, label = "VALID, FOLD {}".format(foldn))
    axes[0, 0].legend()
    axes[0,0].set_xlabel('epoch')
    axes[0,0].set_ylabel('NLLloss')

    # plot accuracies
    axes[0, 1].plot(np.arange(len(tr_accuracies)), tr_accuracies, lw = 1, label = "TRAIN, FOLD {}".format(foldn))
    axes[0, 1].plot(np.arange(len(val_accuracies)), val_accuracies, lw = 1, label = "VALID, FOLD {}".format(foldn))
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('Accuracies')

    # plot AUCS
    axes[1, 0].plot(np.arange(len(tr_auc)), tr_auc, lw=1, label="TRAIN, FOLD {}".format(foldn))
    axes[1, 0].plot(np.arange(len(val_auc)), val_auc, lw=1, label="VALID, FOLD {}".format(foldn))
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('AUC')

    # title the figure with the model's name
    fig.suptitle(MODELFULLNAME)

#Saving conf mat.
with open(os.path.join(RES_path, MODELFULLNAME + "_CONF_MAT.npy"), 'wb') as f:
    np.save(f, total_conf_matrix)

#Adding Conf Mat to figure
plt.sca(axes[1,1])
if clsfID == "MUL":
    conf_df = pd.DataFrame(total_conf_matrix, RFs, RFs)
else:
    conf_df = pd.DataFrame(total_conf_matrix, ["Rest", str(target)], ["Rest", str(target)])

sn.set(font_scale=1)
sn.heatmap(conf_df, annot=True, annot_kws={"size": 9}, fmt='g', cbar=False)
plt.ylabel("True family")
plt.xlabel("Predicted Family")

#Save the whole plot in TRPLOTS
outpath = os.path.join(TRPLOTS_path, MODELFULLNAME + '.png')
plt.savefig(outpath, dpi = 300)


#RES = pd.concat(frames)
#AGG_AUC = metrics.roc_auc_score(y_true = RES.numeral , y_score = RES.yscore)
#RES.to_csv(os.path.join(RES_path, MODELFULLNAME + "_AGG_AUC_{}.scores".format(round(AGG_AUC,4))))
