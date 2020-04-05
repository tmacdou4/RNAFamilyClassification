import numpy as np
import warnings
import pdb
import os
import torch
from torch import nn
import copy
from models import *
from utility import *

# specs = {'n_hid_lyrs': 2, 'output_size' : 2, 'batch_size' : 50, 'HID1N': 8, 'seq_len': 300, 'ARCH': [200,200]}
#
# net = DNN(specs).cuda()
#
# print(net)
#
# #test sequences of shape (num_seq * seq_len) made of torch.long indexes
# sequences = torch.Tensor(np.random.randint(0, 17, size=(50, 300))).cuda()
#
# print(net(sequences))


# seqs = seq_loader("data", "RF00009", "fasta_unaligned.txt")
# seqs_index = seq_to_nt_ids(seqs)
# fixed = pad_to_fixed_length(seqs_index, max_length=440, random="uniform")
#
# print(fixed[:,-1])

datapath = "data"

#RFs =[path for path in os.listdir(datapath) if os.path.isdir(os.path.join(datapath,path))]

#data, labels = load_data_in_df(RFs, datapath = datapath, max_len = 500)

#labels['numeral'] = np.array(labels['RFAM'] == "RF00009", dtype = int)

RFs = [['RF00005', 'RF00009']]

data, labels = load_data(RFs, datapath=datapath, max_len = 500)

print(data)
print(labels)

#seqs = generate_based_on_family("RF00005", datapath=datapath)
