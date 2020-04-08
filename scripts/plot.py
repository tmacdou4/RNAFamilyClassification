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
parser.add_argument('-Mdl', default = 'DNN',  type = str, dest = 'MODELlayout')
# RFAMID = ALL
parser.add_argument('-exclRFAM', dest = 'exclRFAM',  default = [], nargs = '+',  type = str)
args = parser.parse_args()

# set some static variables
DATApath = 'data'
RESpath = os.path.join(args.MODELlayout, 'OUT')
RFs =[path for path in os.listdir(DATApath) if os.path.isdir(os.path.join(DATApath,path))]
RFIDs = np.setdiff1d(RFs, args.exclRFAM)
ARCHs = ["5.5", "1000.1000"]
MODELID = "WD0.2"
CLFID = "BIN"
model_layout = 'DNN'
seed = 1
xval = 5
# generate table
# RFID TASKID ARCHID MODELID FOLDID MODELSPECS TR_ACC TR_AUC TR_L TR_PROC_TIME TST_ACC TST_AUC
frames = []
AGG_AUC = []
ROC = []
n = 0
# cycle through RFIDs
for rfid in RFIDs:
    # TASKS = ['ZP', 'RP', 'NUCSHFLZP', 'FMLM1']
    tasks = ['ZP','RP','NUCSHFL', 'FMLM1']
    for taskid in tasks :
            for ARCHID in ARCHs:
                ytrue = []
                yscores = []
                REF = "_".join([rfid, taskid, ARCHID])
                for FOLDID in range(1, xval + 1):
                    n += 1
                    MODELFULLNAME = "_".join([rfid, CLFID, taskid, ARCHID, MODELID, str(FOLDID)])
                    path = os.path.join(RESpath, MODELFULLNAME + '.csv') 
                    try :
                            # load in test set y scores 
                            f = open(path, 'r') 
                            scores = f.readlines()[1].split(':')[1]
                            yscores.append(np.array(scores.strip().split(','), dtype = float))
                            f.close()
                            # load in test set y true labels
                            labels_path = os.path.join(model_layout, 'SETS','SEED{}_F{}_{}.csv'.format(seed, FOLDID, xval))
                            ytrue.append(np.array(pd.read_csv(labels_path, index_col = 0).numeral))
                            frames.append(pd.read_csv(path, sep = '\t', comment = "#", index_col = 0))
                    except FileNotFoundError :
                            print('FILE :   {} NOT FOUND !'.format(path))
                if len(ytrue) : 
                    yt = np.concatenate(ytrue)
                    ys = np.concatenate(yscores)
                    computed_AUC = metrics.roc_auc_score(y_true = yt, y_score = ys)
                    FPR, TPR, thresholds = metrics.roc_curve(y_true = yt, y_score = ys)
                    ROC.append((REF, {'FPR':FPR, 'TPR':TPR}))
                    AGG_AUC.append(computed_AUC)          

# generate merged dataframe
RESULTS = pd.concat(frames)
# verbose
print ('collected {} / {} frames '.format(RESULTS.shape[0], n))
# compute aggregated AUC from all tests folds

pdb.set_trace()
# group by MODEL
RESULTS = RESULTS.group_by(['RFID', 'TASKID', 'ARCHID'])
# update RESULTS

# set a RES filepath
RESpath = os.path.join('sauves_FIG/RNAClassification/')
util.assert_mkdir(RESpath)
# start plotting
# first TEST AUC scatter by RFid 
fig, axes = plt.subplots(figsize = (5, 5 * len(RFIDs)),nrows = len(RFIDs))
for j, ARCHID in ARCHs:
        for i, ((ref, roc), AUC) in enumerate(zip(ROC,AGG_AUC)):
            axes[i][j].plot(roc['FPR'], roc['TPR'])
            axes[i][j].set_title('REF: {} | AUC: {}'.format(ref, round(AUC, 3)))
# save data table

# save figure
plt.savefig(os.path.join(RESpath, 'ALL_RES.png'), dpi = 300)



