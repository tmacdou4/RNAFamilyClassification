import utility as util
import os
import argparse
import numpy as np
import pandas as pd

# parse arguments
parser = argparse.ArgumentParser()
# MODELlayout = DNN
parser.add_argument('-Mdl', default = 'DNN',  type = str, dest = 'MODELlayout')
# RFAMID = ALL
parser.add_argument('-exclRFAM', dest = 'exclRFAM',  default = [], nargs = '+',  type = str)
args = parser.parse_args()

# set some static variables
RESpath = os.path.join(args.MODELlayout, 'OUT')
RFIDs = np.setdiff(os.listdir(RESpath), args.exclRFAM)
ARCHID = "5.5"
MODELID = "WD0.2"
CLFID = ""
# generate table
# RFID TASKID ARCHID MODELID FOLDID MODELSPECS TR_ACC TR_AUC TR_L TR_PROC_TIME TST_ACC TST_AUC
# cycle through RFIDs
for rfid in RFIDs:
    # TASKS = ['ZP', 'RP', 'NUCSHFLZP', 'FMLM1']
    tasks = ['ZP','RP','NUCSHFL', 'FMLM1']
    for taskid in tasks :
        for FOLDID in range(1,6):
            MODELFULLNAME = "_".join([rfid, CLFID, , taskid, ARCHID, MODELID, FOLDID]
            path = os.path.join(RESpath, MODELFULLNAME + '.csv')  
            df = pd.readcsv(path)
# generate




