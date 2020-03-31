import os
import pandas as pd
import numpy as np
import collections
from scipy import stats
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader

filepath = '../data/'
familypath = 'RF00005/'
filename= 'fasta_unaligned.txt'

# List of all possible NT codes from the wikipedia page about fasta
nt_vocab = ["A", "C", "G", "U", "N", "R", "-", "K", "M", "S", "W", "B", "D", "H", "V", "Y", "T"]

#returns list of sequence strings
def seq_loader(filep, familyp, fname):
    file = open(os.path.join(filep,familyp, fname), "r")
    seqs=[]
    for i, line in enumerate(file):
        if i % 2 == 1:
            #removing newline characters as well
            seqs.append(line.replace("\n", ""))
    return seqs

#returns set of non-standard chars
def non_std_char(seqs):
    #searching for non-standard nucleotides
    chars = set()
    for l in seqs:
        for c in l:
            if c != "A" and c!= "U" and c!= "C" and c!= "G":
                chars.add(c)
    return chars

#returns 2D numpy array of index ints
def pad_to_fixed_length(seqs, max_length = 100):
    for l in seqs:
        if len(l) > max_length:
            max_length = len(l)

    fixed_seqs = np.zeros(shape=(len(seqs), max_length))
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            fixed_seqs[i][j] = seqs[i][j]

        for j in range(len(seqs[i]), (max_length-len(seqs[i]))+len(seqs[i]), 1):
            fixed_seqs[i][j] = 6

    return fixed_seqs

#returns 2D numpy array of index ints
def truncate_to_fixed_length(seqs):
    min_length = 100000
    for i in range(len(seqs)):
        if len(seqs[i]) < min_length:
            min_length = len(seqs[i])

    fixed_seqs = np.zeros(shape=(len(seqs), min_length))
    for i in range(len(seqs)):
        fixed_seqs[i] = seqs[i][:min_length]

    return fixed_seqs

#returns list of list of nucleotide ID integers
def seq_to_nt_ids(seqs):
    nt_to_id = {k: v for v, k in enumerate(nt_vocab)}
    return [[nt_to_id[c] for c in l] for l in seqs]

#returns 3D torch tensor of index ints
def one_hot_encoding(data):
    one_hot_data = torch.zeros(data.shape[0], data.shape[1], len(nt_vocab))
    for i in range(one_hot_data.shape[0]):
        for j in range(one_hot_data.shape[1]):
            one_hot_data[i][j][int(data[i][j])] = 1

    return one_hot_data
#
#
# SOME TORCH CLASSES AND CUSTOM ONES
#
#
class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data['data'][idx]), torch.LongTensor(np.array(self.data['labels'][idx]))
    def __len__(self):
        return self.data['data'].shape[0]

class BalancedDataPicker(Dataset):
        def __init__(self, data):
            self.data = data
            self.size = self.data['data'].shape[0]
        def __getitem__(self, idx):
            # first pick a class at random
            classe = np.random.choice(np.unique(self.data['labels']))
            # then pick a sample in that class 
            idx = np.random.choice(np.where(self.data['labels'] == classe)[0])
            # then return x,y tuple for this index
            return torch.FloatTensor(self.data['data'][idx]), torch.LongTensor(np.array(self.data['labels'][idx])) 
        def __len__(self):
            return self.size

# assert mkdir 
def assert_mkdir(path):
    """
    FUN that takes a path as input and checks if it exists, then if not, will recursively make the directories to complete the path
    """
        
    currdir = ''
    for dir in path.split('/'):
        dir = dir.replace('-','').replace(' ', '').replace('/', '_') 
        if not os.path.exists(os.path.join(currdir, dir)):
            os.mkdir(os.path.join(currdir, dir))
            print(os.path.join(currdir, dir), ' has been created')
        currdir = os.path.join(str(currdir), str(dir))

#composition of a given RFAM family
def generate_based_on_family(RFAM_name, datapath = "../data"):

    #Do some statistics to match an RFAM family in sequence length and sequence composition
    seqs = seq_loader(datapath, RFAM_name, "fasta_unaligned.txt")
    data = seq_to_nt_ids(seqs)

    #match family in sequence length
    lens = []
    for seq in data:
        lens.append(len(seq))
    num_seq = len(lens)
    lens = np.array(lens)
    len_mean = np.mean(lens)
    len_std = np.std(lens)

    #match family in sequence composition
    nt_probs = np.zeros(17)
    for l in data:
        for c in l:
            nt_probs[c] += 1

    scaling_factor = 1.0/sum(nt_probs)
    nt_probs = nt_probs*scaling_factor

    pmf = stats.rv_discrete(values=(list(range(17)), nt_probs))

    #actual random part
    lengths = np.random.normal(loc=len_mean, scale=len_std, size=num_seq)
    lengths = [int(x) for x in list(lengths)]

    rand_seqs = []
    for i in range(len(lengths)):
        rand_seqs.append(list(pmf.rvs(size=lengths[i])))

    return rand_seqs

# loading data into data frame
def load_data_in_df(RFs, datapath = "../data" ,seq_len = 500):
    data = []
    labels = []
    seeds = []
    for RF in RFs:            
        seqs = seq_loader(datapath, RF, 'fasta_unaligned.txt')
        seqs_index = seq_to_nt_ids(seqs)
        fixed_seqs = pad_to_fixed_length(seqs_index, max_length = seq_len) # fix the max manually ? to be fixed
        data.append(fixed_seqs)
        labels.append([RF for i in range(len(fixed_seqs))])
        seeds.append(seqs)
    seeds = np.concatenate(seeds)
    data = np.concatenate(data)    
    labels  = np.concatenate(labels)
    data_df = pd.DataFrame(data, index = seeds)
    labels_df = pd.DataFrame({'RFAM': labels}, index = seeds)
    return data_df, labels_df

seqs = seq_loader(filepath, familypath, filename)
indexes = seq_to_nt_ids(seqs)
#indexes = generate_based_on_family("RF00005")
data = pad_to_fixed_length(indexes)
#data = one_hot_encoding(data)
print(data.shape)


