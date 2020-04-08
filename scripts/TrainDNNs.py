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
from models import *
import pdb
import os
import torch
from sklearn import metrics
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader 

#
# TrainDNNs.py
#

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# parse arguments 
parser.add_argument('-arch', dest = 'ARCH', default = [10, 2], nargs = 2, type = int, help = 'set the architecture of the 1 or 2 - layers DNN model. ex 100 in 1st , 2000 in second is typed : -arch 100 2000')
parser.add_argument('-epochs',dest = 'EPOCHS', default = 100 , type = int, help = 'nb of max epochs')
parser.add_argument('-wd', dest = 'WEIGHT_DECAY', type = float, default = 0.2, help = 'L2 parametrization [0:no regularization]')
parser.add_argument('-xval', dest = 'XVAL', default = 5, type = int, help= 'number of folds for crossvalidation')
parser.add_argument('-seed', dest = 'SEED', default= 1, type = int, help = 'random seed')
parser.add_argument('-d', dest = 'DEVICE', default= 'cuda:0', type = str, help = 'device ex cuda:0')

args = parser.parse_args()

# some other global variables
# paths
datapath = 'data'
# static values 
modelname = 'DNN'
lr = 1e-4
drp=0
vs = 'REST'
bs = 128
loader = 'balanced'
seq_len = 600 # how to get the optimal number efficiently ?

# Set RFs to include
RFs =[path for path in os.listdir(datapath) if os.path.isdir(os.path.join(datapath,path))] 
# TASKS = ['ZP','RP','NUCSHFLZP', 'FMLM1']
# only ZP implemented so far
TASKS = ['ZP']

# generate architectures
maxNodes = args.ARCH[0]
maxLyrs = args.ARCH[1]
minNodes = 5
X = np.array([max(minNodes, 10 ** log10) for log10 in range(int(np.log10(maxNodes)))] )
L = np.arange(2,maxLyrs + 1)
ARCHs = []
for N in L:
    combN = [list(e) for e in it.combinations_with_replacement(X[::-1], N)][::-1]
    for a in combN :
        ARCHs.append(a)
# SET ARCHs
ARCHs = [[5,5],[1000,1000]]
# cycle through TASKS (4):
for taskID in TASKS: 
        # cycle through architectures (1)
        for ARCH  in ARCHs:
            # cycle through RNA families (24) 
            for rfID in RFs: 
                # update some variables
                target = rfID
                modelID = 'WD{}EP{}'.format(args.WEIGHT_DECAY, args.EPOCHS)
                CLSFID = 'BIN'
                
                # loading data into data frame
                data, labels = load_data_in_df(target, RFs, taskID, datapath = datapath, max_len = seq_len)
                np.random.seed(args.SEED) # set numpy seed for data shuffle  
                # multiclass ONLY
                # numeric_labels = dict(zip(np.unique(labels['RFAM']), np.arange(len(np.unique(labels['RFAM'])))))
                # labels['numeral'] = [numeric_labels[l] for l in labels['RFAM']]
                labels['numeral'] = np.array(labels['RFAM'] == target, dtype = int)
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
                SETS_path = os.path.join(modelname, 'SETS')
                MODELSPECS_path = os.path.join(modelname, 'MODELSPECS')
                MODELS_path = os.path.join(modelname, 'MODELS')
                RES_path = os.path.join(modelname, 'OUT')
                TRPLOTS_path = os.path.join(modelname, 'TRPLOTS')
                assert_mkdir(SETS_path)
                assert_mkdir(MODELSPECS_path)
                assert_mkdir(MODELS_path)
                assert_mkdir(RES_path) 
                assert_mkdir(TRPLOTS_path)

                # define train function 
                from train import train
                # init fig, axes for plotting 
                fig, axes = plt.subplots(ncols = 2, figsize = (20,10))
                annotations = "" 
                # foreach fold in xval (5)
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
                        'target': target,
                        'seq_len': seq_len,
                        'input_size': data.shape[1],
                        'batch_size' : bs, # train_size / 10 ,
                        'wd' : args.WEIGHT_DECAY, 
                        'lr': lr,
                        'drp': drp,
                        'ARCHID': '.'.join([str(e) for e in ARCH]),
                        'CLSFID': CLSFID,
                        'TASKID': taskID, 
                        'MODID': modelID,
                        'RFID': rfID, 
                        'FOLDID': foldn, 
                        'lossfn': torch.nn.BCELoss(),
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
                        'tst_l' : float('nan')
                        }
                    print('RFID {} (BINARY) TASK {} ARCH {} MODEL {} fold {} / {}'.format(rfID, taskID, ARCH, modelID, foldn, args.XVAL)) 
                    # store some static values 
                    nsamples = model_specs['nseeds']
                    test_size = model_specs['test_size']
                    IDs = np.array([model_specs['RFID'], model_specs['CLSFID'], model_specs['TASKID'], model_specs['ARCHID'], model_specs['MODID'], foldn],dtype = str )
                    MODELFULLNAME = '_'.join(IDs)
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
                    # store nb of params in model
                    model_specs['n_free_params'] = get_n_params(model)
                    # train model
                    print('training model...')
                    # time stamp
                    startime = time.clock()
                    train(model, dl,  model_specs, device = model_specs['device'], foldn=foldn)
                    # model is trained, record time, prepare to save model under a REFID
                    # save torch model on disk under the name MODELFULLNAME.txt
                    torch.save(model.state_dict(), os.path.join(MODELS_path, '{}.txt'.format(MODELFULLNAME)))
                    # save up some reported values
                    # update model_specs with various reports
                    model_specs['tr_proc_time'] = time.clock() - startime
                    model_specs['ARCH'] = ".".join([str(e) for e in model_specs['ARCH']])
                    # save model_specs dict under the name MODELFULLNAME.specs 
                    with open(os.path.join(MODELSPECS_path, '{}.specs'.format(MODELFULLNAME)), 'w') as o : o.write(str(model_specs)) # to be updated 
                    # test
                    out = model(torch.Tensor(TEST_X.values).cuda(args.DEVICE))
                    acc = out.argmax(dim = -1).detach().cpu().numpy() == TEST_Y.numeral
                    model_specs['tst_acc'] = float(acc.mean())
                    if len(np.unique(TEST_Y.numeral)) > 1: model_specs['tst_auc'] = metrics.roc_auc_score(y_true = TEST_Y.numeral, y_score = out[:,1].detach().cpu().numpy())
                    else: model_specs['tst_auc'] = float('nan') 
                    # report Training in outfile
                    currDF =  pd.DataFrame(list(model_specs.values()), index = list(model_specs.keys())).T
                    # create outfile 
                    outFile = open(os.path.join(RES_path, MODELFULLNAME + ".csv"), 'w')
                    # insert comments
                    outFile.write("##" + str(args) + "\n")
                    outFile.write("## test yscores:" + ",".join([str(score) for score in out[:,1].detach().cpu().numpy()]) + "\n")    
                    currDF.to_csv(outFile, sep = '\t')
                    outFile.close()
                    # plot training curves and final accuracies on test
                    nbsteps = len(model_specs['tr_l'])
                    skip =10 
                    losses = model_specs['tr_l'][np.arange(1,nbsteps,skip)]
                    accuracies = model_specs['tr_acc'][np.arange(1, nbsteps,skip)]
                    aucs = model_specs['tr_auc'][np.arange(1, nbsteps,skip)]
                    model_specs['tr_l'] = model_specs['tr_l'][-1]
                    model_specs['tr_acc'] = model_specs['tr_acc'][-1]
                    model_specs['tr_auc'] = model_specs['tr_auc'][-1]
                    # PLOT RESULTS 
                    # plot | tr_l | tr_acc | tr_auc
                    axes[0].plot(np.arange(len(losses)) * skip, losses, lw = 1, c = 'grey')
                    axes[1].plot(np.arange(len(aucs)) * skip, aucs, lw = 1, c = 'b')
                    # annotate last tr_l, _tr_acc, tr_auc, tr_proc_time
                    auc_annotm = [str(e).zfill(5) for e in np.round([model_specs['tr_acc'], model_specs['tr_auc'] , model_specs['tst_acc'], model_specs['tst_auc']], 3)]
                    loss_annotm = [str(e).zfill(5) for e in np.round([model_specs['tr_l'], model_specs['tr_proc_time'] , model_specs['tst_l']], 3)]
                    
                    auc_annots = "FOLD NB {} | TR. ACC: {} | TR. AUC:{} | TST. ACC: {} | TST. AUC: {}".format(foldn,auc_annotm[0], auc_annotm[1], auc_annotm[2], auc_annotm[3])
                    loss_annots = "FOLD NB {} | TR. LOSS: {} | TR. PROC TIME:{} | TST. L: {}".format(foldn,loss_annotm[0], loss_annotm[1], loss_annotm[2])
                    # annotate/scatter tst_l, tst_acc, tst_auc
                    axes[0].scatter(x = [nbsteps], y = [model_specs['tr_l']], s = 5, label = loss_annots)
                    axes[1].scatter(x = [nbsteps], y = [model_specs['tr_auc']], s = 5,  c= 'b')
                    axes[1].scatter(x = [nbsteps], y = [model_specs['tst_auc']], s = 5, label = auc_annots)
                    axes[0].legend()
                    axes[1].legend()
                    # xlabel = gradient steps
                    axes[0].set_xlabel('gradient steps') 
                    axes[0].set_ylabel('BCEloss') 
                    axes[1].set_xlabel('gradient steps') 
                    axes[1].set_ylabel('auc') 
                    # title the figure with REFID
                    REF ="_".join(np.array([ model_specs['RFID'], model_specs['ARCHID'], model_specs['MODID']], dtype = str))
                    fig.suptitle(REF)
                    # make outpath 
                    outpath = os.path.join(TRPLOTS_path, REF + '.png')
                    plt.savefig(outpath, dpi = 300)
