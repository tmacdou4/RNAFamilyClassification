import numpy as np
import warnings
from utility import *
import pdb
import os
import torch
from torch import nn
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DNN(nn.Module):
    def __init__(self, model_specs):
        super(DNN, self).__init__()
        self.num_layers = 2
        self.non_linearity = nn.ReLU()
        self.nt_vocab_size = 17
        self.input_size =


        self.layers = nn.ModuleList()
        #first layer
        self.layers.append(nn.Linear(model_specs['input_size'], model_specs['HID1N']))
        #middle layers
        self.layers.extend(clones(nn.Linear(model_specs['HID1N'], model_specs['HID1N']), self.num_layers-1))
        #output layers
        self.layers.append(nn.Linear(model_specs['HID1N'], model_specs['output_size']))
        self.out_nl = nn.Softmax(dim=-1)

    def forward(self, x):


        for layer in range(self.num_layers):
            # Calculate the hidden states
            # And apply the activation function tanh on it
            hidden[layer] = torch.tanh(self.layers[layer](torch.cat([input_, hidden[layer]], 1)))
            # Apply dropout on this layer, but not for the recurrent units
            input_ = self.dropout(hidden[layer])

        return out

class CNN(nn.Module):
    def __init__(self, model_specs):
        super(CNN, self).__init__()
        pass

    def forward(self, x):
        pass

class RNN(nn.Module):
    def __init__(self, model_specs):
        super(RNN, self).__init__()
        self.emb_size = 2
        self.embeddings = nn.Embedding(17, self.emb_size)
        self.rnn = nn.RNN()
        self.linear = nn.Linear()


    def forward(self, x):

        if inputs.is_cuda:
            device = inputs.get_device()
        else:
            device = torch.device("cpu")

        embed_out = self.embeddings(inputs)
        pass

class LSTM(nn.Module):
    def __init__(self, model_specs):
        super(LSTM, self).__init__()
        self.emb_size = 2
        self.embeddings = nn.Embedding(17, self.emb_size)
        self.rnn = nn.RNN()
        self.linear = nn.Linear()

    def forward(self, x):
        pass