import numpy as np
import warnings
import pdb
import os
from utility import *
import torch
from torch import nn
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DNN(nn.Module):
    def __init__(self, model_specs):
        super(DNN, self).__init__()
        self.batch_size = model_specs['batch_size']

        #number of hidden layers
        self.num_layers = model_specs['n_hid_lyrs']
        self.non_linearity = nn.ReLU()
        self.nt_vocab_size = 17
        self.hidden_size = model_specs['HID1N']
        self.out_size = model_specs['output_size']
        self.seq_len = 10

        #Embbed or one-hot?
        self.embed = False

        if self.embed:
            self.emb_size = 5
            self.embeddings = nn.Embedding(17, self.emb_size)
        else:
            #one-hot encodings are effectively an embedding to space of 17
            self.emb_size = 17

        self.layers = nn.ModuleList()
        #first layer
        self.layers.append(nn.Linear(self.emb_size*self.seq_len, self.hidden_size))
        #middle layers
        self.layers.extend(clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers-1))
        #output layers
        self.out_layer = (nn.Linear(self.hidden_size, self.out_size))
        self.out_nl = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.embed:
            input = self.embeddings(x)
        else:
            input = one_hot_encoding(x)

        input = input.view(self.batch_size, self.seq_len * self.emb_size)
        for layer in range(self.num_layers):
            input = self.non_linearity(self.layers[layer](input))
        input = self.out_layer(input)
        input = self.out_nl(input)
        return input

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