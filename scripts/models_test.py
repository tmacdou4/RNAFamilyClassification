import numpy as np
import warnings
import pdb
import os
import torch
from torch import nn
import copy
from models import *
from utility import *

specs = {'n_hid_lyrs': 2, 'output_size' : 1, 'batch_size' : 50, 'HID1N': 8, 'seq_len': 10}

net = DNN(specs)

#test sequences of shape (num_seq * seq_len) made of torch.long indexes
sequences = torch.tensor(np.random.randint(0, 17, size=(50, 10)), dtype=torch.long)

print(net(sequences).shape)
