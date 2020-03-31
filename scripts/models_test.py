import numpy as np
import warnings
import pdb
import os
import torch
from torch import nn
import copy
from models import *


net = DNN()

sequences = torch.tensor(np.random.randint(0,17,size=(50,10)), dtype=torch.long)

print(net(sequences).shape)