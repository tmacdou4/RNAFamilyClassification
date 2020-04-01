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

        self.seq_len = model_specs['seq_len'] # this is eq to input size !
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

        if x.is_cuda:
            device = x.get_device()
        else:
            device = torch.device("cpu")

        if self.embed:
            input = self.embeddings(x)
        else:
            input = one_hot_encoding(x)
        input = input.view(input.shape[0], self.seq_len * self.emb_size)
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
        self.batch_size = model_specs['batch_size']

        # number of hidden layers
        self.num_layers = model_specs['n_hid_lyrs']
        self.non_linearity = nn.ReLU()
        self.nt_vocab_size = 17
        self.hidden_size = model_specs['HID1N']
        self.out_size = model_specs['output_size']
        self.bidirectional = False
        self.dropout = 0

        self.seq_len = model_specs['seq_len']  # this is eq to input size !
        # Embbed or one-hot?
        self.embed = False

        if self.embed:
            self.emb_size = 5
            self.embeddings = nn.Embedding(17, self.emb_size)
        else:
            # one-hot encodings are effectively an embedding to space of 17
            self.emb_size = 17

        self.rnn = nn.RNN(self.emb_size, self.hidden_size, self.num_layers,
                          batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.out_layer = (nn.Linear(2*self.hidden_size, self.out_size))
        else:
            self.out_layer = (nn.Linear(self.hidden_size, self.out_size))

        self.out_nl = nn.Softmax(dim=-1)


    def forward(self, input):
        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        if self.embed:
            input = self.embeddings(input)
        else:
            input = one_hot_encoding(input)

        _, input = self.rnn(input)

        input = input.view(self.num_layers, int(self.bidirectional)+1, self.batch_size, self.hidden_size)

        input = input[-1]
        if self.bidirectional:
            input = torch.cat((input[0], input[1]), dim=-1)

        input=input.squeeze(0)
        input = self.out_layer(input)
        input = self.out_nl(input)

        return input

class LSTM(nn.Module):
    def __init__(self, model_specs):
        super(LSTM, self).__init__()
        self.batch_size = model_specs['batch_size']

        # number of hidden layers
        self.num_layers = model_specs['n_hid_lyrs']
        self.non_linearity = nn.ReLU()
        self.nt_vocab_size = 17
        self.hidden_size = model_specs['HID1N']
        self.out_size = model_specs['output_size']
        self.bidirectional = False
        self.dropout = 0

        self.seq_len = model_specs['seq_len']  # this is eq to input size !
        # Embbed or one-hot?
        self.embed = False

        if self.embed:
            self.emb_size = 5
            self.embeddings = nn.Embedding(17, self.emb_size)
        else:
            # one-hot encodings are effectively an embedding to space of 17
            self.emb_size = 17

        self.rnn = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers,
                          batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.out_layer = (nn.Linear(2 * self.hidden_size, self.out_size))
        else:
            self.out_layer = (nn.Linear(self.hidden_size, self.out_size))

        self.out_nl = nn.Softmax(dim=-1)

    def forward(self, input):
        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        if self.embed:
            input = self.embeddings(input)
        else:
            input = one_hot_encoding(input)

        #Not sure if should be taking hidden or cell state!
        _, (input, _) = self.rnn(input)

        input = input.view(self.num_layers, int(self.bidirectional)+1, self.batch_size, self.hidden_size)

        input = input[-1]
        if self.bidirectional:
            input = torch.cat((input[0], input[1]), dim=-1)

        input=input.squeeze(0)
        input = self.out_layer(input)
        input = self.out_nl(input)

        return input
