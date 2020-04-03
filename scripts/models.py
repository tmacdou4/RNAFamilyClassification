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

def conv_layer_size(l_in, k, stride, padding=0, dilation=1):
    return int(((l_in + (2*padding) - (dilation*(k-1)) - 1)/stride)+1)

# returns 3D torch tensor of index ints
class One_Hot_Encoding(nn.Module):
    def __init__(self, nt_vocab_size):
        super(One_Hot_Encoding, self).__init__()
        self.nt_vocab_size = nt_vocab_size

    def forward(self, input):

        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        input = input.unsqueeze(2)
        one_hot_data = torch.zeros(input.shape[0], input.shape[1], self.nt_vocab_size).to(device)
        one_hot_data.scatter_(2, input, 1)

        return one_hot_data

class DNN(nn.Module):
    def __init__(self, model_specs):
        super(DNN, self).__init__()
        self.batch_size = model_specs['batch_size']

        #number of hidden layers
        self.num_layers = model_specs['n_hid_lyrs']
        self.non_linearity = nn.ReLU()
        self.nt_vocab_size = 17
        #self.hidden_size = model_specs['hidden_size']
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
            self.embeddings = One_Hot_Encoding(17)

        self.ARCH = [self.emb_size*self.seq_len] + model_specs['ARCH']
        self.layers = nn.ModuleList()
        #first layer
        #self.layers.append(nn.Linear(self.emb_size*self.seq_len, self.ARCH[0]))
        #middle layers
        for i in range(1, self.num_layers+1):
            self.layers.append(nn.Linear(self.ARCH[i-1], self.ARCH[i]))
        #self.layers.extend(clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers-1))
        #output layers
        self.out_layer = (nn.Linear(self.ARCH[-1], self.out_size))

        if self.out_size != 1:
            self.out_nl = nn.Softmax(dim=-1)
        else:
            self.out_nl = nn.Sigmoid()

    def forward(self, input):

        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        input = input.long()

        input = self.embeddings(input)

        input = input.view(input.shape[0], self.seq_len * self.emb_size)

        for layer in range(self.num_layers):
            input = self.non_linearity(self.layers[layer](input))
        input = self.out_layer(input)
        input = self.out_nl(input)
        return input

class CNN(nn.Module):
    def __init__(self, model_specs):
        super(CNN, self).__init__()
        self.batch_size = model_specs['batch_size']
        self.nt_vocab_size = 17
        self.out_size = model_specs['output_size']
        self.seq_len = model_specs['seq_len']  # this is eq to input size !
        self.embed = True # Embed or one-hot?

        if self.embed:
            self.emb_size = 5
            self.embeddings = nn.Embedding(17, self.emb_size)
        else:
            # one-hot encodings are effectively an embedding to space of 17
            self.emb_size = 17
            self.embeddings = One_Hot_Encoding(17)

        #convolutional architecture details
        #1 convolutional layer
        self.num_k = 128
        self.k_size = 22
        self.m_p_size_1 = 100
        self.non_linearity = nn.ReLU()
        self.fc_hidden_size = 500

        self.conv1 = nn.Conv1d(self.emb_size, self.num_k, self.k_size, stride=2)
        self.pool1 = nn.MaxPool1d(self.m_p_size_1, stride=4)

        len_conv = conv_layer_size(self.seq_len, self.k_size, 2)
        len_pool = conv_layer_size(len_conv, self.m_p_size_1, 4)

        self.fc = nn.Linear((self.num_k * len_pool), self.fc_hidden_size)
        self.out_layer = nn.Linear(self.fc_hidden_size, self.out_size)

        if self.out_size != 1:
            self.out_nl = nn.Softmax(dim=-1)
        else:
            self.out_nl = nn.Sigmoid()

    def forward(self, input):
        curr_batch_size = input.size(0)

        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        input = input.long()
        input = self.embeddings(input)

        #input received as (batch_size x sequence_len x embedding_size)
        input = input.transpose(-2, -1)
        #input now (batch_size x embedding size x seq_len)

        input = self.conv1(input)

        input = self.non_linearity(input)
        input = self.pool1(input)

        input = input.view(curr_batch_size, -1)

        input = self.fc(input)
        input = self.non_linearity(input)
        input = self.out_layer(input)
        input = self.out_nl(input)
        return input

class CNN_2L(nn.Module):
    def __init__(self, model_specs):
        super(CNN_2L, self).__init__()
        self.batch_size = model_specs['batch_size']
        self.nt_vocab_size = 17
        self.out_size = model_specs['output_size']
        self.seq_len = model_specs['seq_len']  # this is eq to input size !
        self.embed = True # Embed or one-hot?

        if self.embed:
            self.emb_size = 5
            self.embeddings = nn.Embedding(17, self.emb_size)
        else:
            # one-hot encodings are effectively an embedding to space of 17
            self.emb_size = 17
            self.embeddings = One_Hot_Encoding(17)

        #convolutional architecture details
        #2 convolutional layer
        self.num_k_1 = 128
        self.k_size_1 = 22
        self.m_p_size_1 = 101
        self.num_k_2 = 256
        self.k_size_2 = 22
        self.m_p_size_2 = 10

        self.non_linearity = nn.ReLU()

        self.conv1 = nn.Conv1d(self.emb_size, self.num_k_1, self.k_size_1, stride=2)
        self.pool1 = nn.MaxPool1d(self.m_p_size_1, stride=1)
        self.conv2 = nn.Conv1d(self.num_k_1, self.num_k_2, self.k_size_2, stride=2)
        self.pool2 = nn.MaxPool1d(self.m_p_size_2, stride=1)

        #Calculate the final layer size
        len_conv_1 = conv_layer_size(self.seq_len, self.k_size_1, 2)
        len_pool_1 = conv_layer_size(len_conv_1, self.m_p_size_1, 1)
        len_conv_2 = conv_layer_size(len_pool_1, self.k_size_2, 2)
        len_pool_2 = conv_layer_size(len_conv_2, self.m_p_size_2, 1)

        self.out_layer = nn.Linear((self.num_k_2 * len_pool_2), self.out_size)

        if self.out_size != 1:
            self.out_nl = nn.Softmax(dim=-1)
        else:
            self.out_nl = nn.Sigmoid()

    def forward(self, input):

        curr_batch_size = input.size(0)

        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        input = input.long()
        input = self.embeddings(input)


        #input received as (batch_size x sequence_len x embedding_size)
        input = input.transpose(-2, -1)
        #input now (batch_size x embedding size x seq_len)

        input = self.conv1(input)

        input = self.non_linearity(input)
        input = self.pool1(input)


        input = self.conv2(input)

        input = self.non_linearity(input)
        input = self.pool2(input)

        input = input.view(curr_batch_size, -1)

        input = self.out_layer(input)
        input = self.out_nl(input)
        return input

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
        self.bidirectional = True
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
            self.embeddings = One_Hot_Encoding(17)

        self.rnn = nn.RNN(self.emb_size, self.hidden_size, self.num_layers,
                          batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.out_layer = (nn.Linear(2*self.hidden_size, self.out_size))
        else:
            self.out_layer = (nn.Linear(self.hidden_size, self.out_size))

        if self.out_size != 1:
            self.out_nl = nn.Softmax(dim=-1)
        else:
            self.out_nl = nn.Sigmoid()


    def forward(self, input):
        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        input = input.long()
        input = self.embeddings(input)

        _, input = self.rnn(input)

        #the -1 in this view woudl be batch size, so this accounts for non-standard batch sizes
        input = input.view(self.num_layers, int(self.bidirectional)+1, -1, self.hidden_size)

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
        self.bidirectional = True
        self.dropout = 0

        self.seq_len = model_specs['seq_len']  # this is eq to input size !
        # Embbed or one-hot?
        self.embed = True

        if self.embed:
            self.emb_size = 5
            self.embeddings = nn.Embedding(17, self.emb_size)
        else:
            # one-hot encodings are effectively an embedding to space of 17
            self.emb_size = 17
            self.embeddings = One_Hot_Encoding(17)

        self.rnn = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers,
                          batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.out_layer = (nn.Linear(2 * self.hidden_size, self.out_size))
        else:
            self.out_layer = (nn.Linear(self.hidden_size, self.out_size))

        if self.out_size != 1:
            self.out_nl = nn.Softmax(dim=-1)
        else:
            self.out_nl = nn.Sigmoid()

    def forward(self, input):
        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device("cpu")

        input = input.long()

        input = self.embeddings(input)


        #Not sure if should be taking hidden or cell state!
        _, (_, input) = self.rnn(input)

        # the -1 in this view would be batch size, so this accounts for non-standard batch sizes
        input = input.view(self.num_layers, int(self.bidirectional)+1, -1, self.hidden_size)

        input = input[-1]
        if self.bidirectional:
            input = torch.cat((input[0], input[1]), dim=-1)

        input=input.squeeze(0)
        input = self.out_layer(input)
        input = self.out_nl(input)

        return input
