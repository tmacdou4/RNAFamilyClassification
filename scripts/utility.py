import os
import pandas as pd
import numpy as np
import collections
filepath = 'data/'
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

#returns a list of strings, padded
def pad_to_fixed_length(seqs, max_length = 100):
    for l in seqs:
        if len(l) > max_length:
            max_length = len(l)

    fixed_seqs = []
    for i, l in enumerate(seqs):
        fixed_seqs.append(l)
        for j in range(max_length-len(l)):
            fixed_seqs[i] = fixed_seqs[i] + "-"

    return fixed_seqs

#returns a list of strings, truncated
def truncate_to_fixed_length(seqs):
    min_length = 100000
    for l in seqs:
        if len(l) < min_length:
            min_length = len(l)

    fixed_seqs = []
    for l in seqs:
        fixed_seqs.append(l[:min_length])

    return fixed_seqs

#returns np array of nucleotide index integers
def flatten(data):
    return np.array([c for l in data for c in l], dtype=np.int32)

#returns list of list of nucleotide ID integers
def seq_to_nt_ids(seqs):
    nt_to_id = {k: v for v, k in enumerate(nt_vocab)}
    return [[nt_to_id[c] for c in l] for l in seqs]

#returns a list of list of one-hot encoded numpy vectors
def one_hot_encoding(data):
    one_hot_data = []
    for i, l in enumerate(data):
        one_hot_data.append([])
        for j, index in enumerate(l):
            one_hot_data[i].append(np.zeros(len(nt_vocab)))
            one_hot_data[i][j][index] = 1

    return one_hot_data


# loading data into data frame
def load_data_in_df(RFs, datapath = 'data',seq_len = 500):
    data = []
    labels = []
    seeds = []
    for RF in RFs:            
        seqs = seq_loader(datapath,RF, 'fasta_unaligned.txt')
        fixed_seqs = pad_to_fixed_length(seqs, max_length = seq_len) # fix the max manually ? to be fixed
        seqs_onehot = seq_to_nt_ids(fixed_seqs)
        data.append(seqs_onehot)
        labels.append([RF for i in range(len(seqs_onehot))])
        seeds.append(seqs)
    seeds = np.concatenate(seeds)
    data = np.concatenate(data)    
    labels  = np.concatenate(labels)
    data_df = pd.DataFrame(data, index = seeds)
    labels_df = pd.DataFrame({'RFAM': labels}, index = seeds)
    return data_df, labels_df

seqs = seq_loader(filepath, familypath, filename)
fixed_seqs = pad_to_fixed_length(seqs)

data = seq_to_nt_ids(fixed_seqs)
#print(one_hot_encoding(data))
