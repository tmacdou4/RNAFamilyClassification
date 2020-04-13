import matplotlib as mpl
mpl.use('Agg')
import utility as util
import os
import argparse
import numpy as np
import pandas as pd
import pdb
from io import StringIO
import matplotlib.pyplot as plt
from sklearn import metrics 

# parse arguments
parser = argparse.ArgumentParser()
# MODELlayout = DNN
parser.add_argument('-models',dest = 'MODELS', default = None, nargs = "+",  type = str, help = 'model training output  filenames')
parser.add_argument('-rfs',dest = 'RFS', default = None, nargs = "+",  type = str, help = 'RFAM IDs ')
parser.add_argument('-seed', default = 1, dest = 'SEED', type = int,help = 'seed used for split')
parser.add_argument('-xval', default = None, dest = 'XVAL',  type =int, help = 'crossval')
args = parser.parse_args()


# set some static variables
DATApath = 'data'
MODELSPECS_path = os.path.join('DNN', 'MODELSPECS')
RESpath = os.path.join('DNN', 'OUT')
xval = args.XVAL
seed = args.SEED
RFs_ROC = []
# generate table
# RFID TASKID ARCHID MODELID FOLDID MODELSPECS TR_ACC TR_AUC TR_L TR_PROC_TIME TST_ACC TST_AUC
frames = []
n = 0
for RF in args.RFS:
        ROC = []
        for model in args.MODELS:
                # init arrays
                ytrue = []
                yscores = []
                # cycle through modelnames
                for FOLDID in range(1, xval + 1):
                    n += 1
                    MODELFULLNAME = "_".join([RF, model, str(FOLDID)])
                    path = os.path.join(RESpath, MODELFULLNAME + '.csv') 
                    try :
                            # load in test set y scores, tr losses, tr accs, tr aucs 
                            f = open(path, 'r') 
                            lines = f.readlines()
                            scores = np.array(lines[1].strip().split(':')[-1].split(','), dtype = float)
                            # tr_losses = np.array(lines[2].strip().split(':')[-1].split(','), dtype = float)
                            # tr_aucs = np.array(lines[3].strip().split(':')[-1].split(','), dtype = float)
                            # tr_accs = np.array(lines[4].strip().split(':')[-1].split(','), dtype = float)
                            
                            yscores.append(scores)
                            f.close()
                            # load in test set y true labels
                            labels_path = os.path.join('DNN', 'SETS', 'SEED{}_F{}_{}.csv'.format(seed, FOLDID, xval))
                            ytrue.append(np.array(pd.read_csv(labels_path, index_col = 0).numeral))
                            frames.append(pd.read_csv(path, sep = '\t', comment = "#", index_col = 0))
                    except FileNotFoundError :
                            print('FILE :   {} NOT FOUND !'.format(path))
                if len(ytrue) : 
                    yt = np.concatenate(ytrue)
                    ys = np.concatenate(yscores)
                    computed_AUC = metrics.roc_auc_score(y_true = yt, y_score = ys)
                    FPR, TPR, thresholds = metrics.roc_curve(y_true = yt, y_score = ys)
                    ROC.append((model, {'FPR':FPR, 'TPR':TPR, 'AGG_AUC' : computed_AUC}))          
        RFs_ROC.append((RF,ROC))
# generate merged dataframe
RESULTS = pd.concat(frames)
pdb.set_trace()
# verbose
print ('collected {} / {} frames '.format(RESULTS.shape[0], n))
# compute aggregated AUC from all tests folds

# group by MODEL
# RESULTS = RESULTS.group_by(['RFID', 'TASKID', 'ARCHID'])
# update RESULTS

# set a RES filepath
RESpath = os.path.join('sauves_FIG/RNAClassification/')
util.assert_mkdir(RESpath)
# start plotting
# first TEST AUC scatter by RFid 
fig, axes = plt.subplots(figsize = (15,30), ncols = 6 , nrows = 4)
for i, (RF, (model, roc_dict)) in enumerate(RFs_ROC):
    axes.ravel()[i].plot(roc_dict['FPR'], roc_dict['TPR'], label = '{} | AGG_AUC : {}'.format(model, round(roc_dict['AGG_AUC'], 3)) ,  lw = 2)
    axes.ravel()[i].set_title('REF: {}'.format(RF))
    axes.ravel()[i].set_xlabel('FPR')
    axes.ravel()[i].set_ylabel('TPR')
    axes.ravel()[i].legend('TPR')
# save data table

# save figure
plt.savefig(os.path.join('/u/sauves/public_html/fig/RNAClassification/', 'ALL_RES.png'), dpi = 300)



