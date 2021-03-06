import os
import pandas as pd
import numpy as np
import collections
from scipy import stats
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import itertools
import random




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

#Takes a list of list of ints
#returns 2D numpy array of index ints
#padded up to max_length with either the "-" character
#(for random = "none"), uniform across ACGU for random = "uniform"
#Also now truncates all sequences longer than max_length
def pad_to_fixed_length(seqs, max_length = 100, random="none"):
    # For now, not using the structure that finds the max length sequence
    # for l in seqs:
    #     if len(l) > max_length:
    #         max_length = len(l)

    fixed_seqs = 6*np.ones(shape=(len(seqs), max_length))

    for i in range(len(seqs)):
        if len(seqs[i]) < max_length:
            fixed_seqs[i][:len(seqs[i])] = np.array(seqs[i])
        else:
            fixed_seqs[i] = np.array(seqs[i])[:max_length]
    if random == "none":
        #sequence already padded with "6's" which correspond to the "-" nucleotide
        pass

    elif random == "uniform":
        for i in range(len(seqs)):
            if len(seqs[i]) < max_length:
                fixed_seqs[i][len(seqs[i]):] = np.random.randint(0, 5, size=(1,max_length-len(seqs[i])))

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
            return torch.FloatTensor(self.data['data'][idx]), torch.FloatTensor(np.array(self.data['labels'][idx])) 
        def __len__(self):
            return self.size

class ValidationDataPicker(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = self.data['data'].shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data['data'][idx]), torch.FloatTensor(np.array(self.data['labels'][idx]))

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
def generate_based_on_family(RFAM_name, datapath = "../data", order=0):

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

# Only does first and second order. Takes an RFAM name
# and returns a list of list of nucleotide id's where each one is the same length
# the the corresponding real sequence but the content is randomized with a 0th or
# 1st order markov method. Only considers the 4 main base pairs and discards the rest.
# (They're not that common in this dataset although they do appear)
def markov_generate(RFAM_name, datapath = "../data", order=1):
    seqs = seq_loader(datapath, RFAM_name, "fasta_unaligned.txt")
    data = seq_to_nt_ids(seqs)

    #data = RFAM_name

    nt_vocab_size = 4

    prob_grid_0 = np.zeros(nt_vocab_size)
    prob_grid_1 = np.zeros((nt_vocab_size, nt_vocab_size))
    prob_grid_2 = np.zeros((nt_vocab_size, nt_vocab_size, nt_vocab_size))

    for l in data:
        prev_1 = -1
        prev_2 = -1

        for c in l:
            if c < 4:
                prob_grid_0[c] += 1
                if prev_1 >= 0:
                    prob_grid_1[prev_1][c] += 1
                if prev_2 >= 0:
                    prob_grid_2[prev_2][prev_1][c] += 1

                prev_2 = prev_1
                prev_1 = c

    prob_grid_0 = prob_grid_0/np.sum(prob_grid_0)

    sum_1 = np.sum(prob_grid_1, axis=-1).reshape(prob_grid_1.shape[0],-1)

    prob_grid_1 = prob_grid_1/sum_1

    pmf_0 = stats.rv_discrete(values=(list(range(nt_vocab_size)), prob_grid_0))

    pmf_1 = []
    for i in range(nt_vocab_size):
        pmf_1.append(stats.rv_discrete(values=(list(range(nt_vocab_size)), prob_grid_1[i])))

    lengths = [len(x) for x in seqs]

    rand_seqs = []

    for i, l in enumerate(lengths):
        rand_seqs.append([])
        rand_seqs[i].append(pmf_0.rvs())

        if order == 0:
            for j in range(1,l):
                rand_seqs[i].append(pmf_0.rvs())

        elif order == 1:
            for j in range(1,l):
                rand_seqs[i].append(pmf_1[rand_seqs[i][j-1]].rvs())

    return rand_seqs

def shuffle_seqs_in_family(RFAM_name, datapath = "../data"):
    seqs = seq_loader(datapath, RFAM_name, "fasta_unaligned.txt")
    data = seq_to_nt_ids(seqs)
    shuffled_data = [random.sample(x, len(x)) for x in data]

    return shuffled_data

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# loading data into data frame
def load_data_in_df(target, RFs, method, datapath = "../data" ,max_len = 500):
    data = []
    labels = []
    seeds = []
    for RF in RFs:            
        seqs = seq_loader(datapath, RF, 'fasta_unaligned.txt')
        seqs_index = seq_to_nt_ids(seqs)
        
        # zero padding method
        if method == 'ZP': 
                fixed_seqs = pad_to_fixed_length(seqs_index, max_length = max_len) # fix the max manually ? to be fixed
                data.append(fixed_seqs)
                labels.append([RF for i in range(len(fixed_seqs))])
                seeds.append(seqs)

        # random padding method
        elif method == 'RP':
                fixed_seqs = pad_to_fixed_length(seqs_index, max_length = max_len, random="uniform")
                data.append(fixed_seqs)
                labels.append([RF for i in range(len(fixed_seqs))])
                seeds.append(seqs)

        # nucshfl + zero padding method
        # not sure what method this is supposed to be, so I added 2 options
        # generated_based_on_family -> samples a sequence length from gaussian,
        # then sample nucleotides to fill the sequences from the pmf of composition for that family
        # shuffle_seqs_in_family -> for each sequence in the family, shuffle the sequence
        elif method == 'NUCSHFLZP':
                #seqs_index = generate_based_on_family(RF, datapath=datapath)
                seqs_index = shuffle_seqs_in_family(RF, datapath=datapath)
                fixed_seqs = pad_to_fixed_length(seqs_index, max_length = max_len)
                data.append(fixed_seqs)
                labels.append([RF for i in range(len(fixed_seqs))])
                seeds.append(seqs)

        elif method == 'NUCSHFLRP':
                seqs_index = shuffle_seqs_in_family(RF, datapath=datapath)
                fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len, random="uniform")
                data.append(fixed_seqs)
                labels.append([RF for i in range(len(fixed_seqs))])
                seeds.append(seqs)

        # markov chain random generator order 1 + zero padding (  within target family only ! ) 
        elif method == 'DINUCSHFL' and RF == target:

                fixed_seqs = pad_to_fixed_length(seqs_index, max_length= max_len)

                rand_seqs_index = markov_generate(target, datapath="data", order=1)
                rand_fixed_seqs = pad_to_fixed_length(rand_seqs_index, max_length=max_len)

                seeds = np.concatenate([seqs,["RAND" for _ in range(len(rand_fixed_seqs))]])
                data = np.concatenate([fixed_seqs,rand_fixed_seqs])
                labels = np.concatenate([[RF for i in range(len(fixed_seqs))], ['RANDOM_DINUCSHFL' for _ in range(len(fixed_seqs)) ]])
                return pd.DataFrame(data), pd.DataFrame({'RFAM': labels, 'seed': seeds})
    
    seeds = np.concatenate(seeds)
    data = np.concatenate(data)    
    labels  = np.concatenate(labels)
    data_df = pd.DataFrame(data)
    labels_df = pd.DataFrame({'RFAM': labels, 'seed': seeds})
    return data_df, labels_df

#Takes an RF structure list (see below), a datapath string and a max_len int
#Returns 2 data frames : one for data, with indexes and NT id's and one for
#Labels, with RFAM ID ["RFAM"], Seed sequence, ["seed"] and class label, "numeral"

#The "RF structure list" determines which families are included and which labels
#they are assigned.

#RF_structure = ['ALL'] is a completely multiclass setup

#RF_structure = ['RF00005', 'REST'] is one vs rest with RF00005 as the one out.

#RF_structure = ['RF00005', 'RANDOM'] is one vs one with RF00005 and an equal number
#of randomly generated sequences based on RF00005. If more families are included, the random
#sequences are always based on the first listed sequence.

#RF_structure = ['RF00005', 'RF00009'] is one vs one between these classes

#RF_structure = ['RF00005', 'RF00009', 'REST'] is three way classification between 2
#specific classes and the third class being the rest of the sequences in one big class

#RF_structure = [['RF00005', 'RF00009'], 'REST'] is two way classification between RF00005 and RF00009
#combined into one class, and the rest of the sequences in one class

##RF_structure = [['RF00005', 'RF00009'], ['RF00001', 'RF01865'], 'RF01852'] is three way
# classification between RF00005 and RF00009 combined into one class,
# RF00001 and RF01865 combined into one class, and RF01865 as the third class

#All of the other combinations of these are possible
def load_data(target, RF_structure, taskID, datapath="data", max_len=500):
    data = []
    labels = []
    labels_numeral = []
    seeds = []
    seen_fams = set()
    first_fam = None

    all_RFs = [path for path in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, path))]

    for i, klass in enumerate(RF_structure):
        if type(klass) == list:
            for fam in klass:
                if first_fam is None:
                    first_fam = fam
                if fam not in seen_fams:
                    seen_fams.add(fam)

                    #call single family loader
                    seqs = seq_loader(datapath, fam, 'fasta_unaligned.txt')
                    seqs_index = seq_to_nt_ids(seqs)
                    fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len)

                    data.append(fixed_seqs)
                    labels.append([fam for _ in range(len(fixed_seqs))])
                    labels_numeral.append([i for _ in range(len(fixed_seqs))])
                    seeds.append(seqs)

        elif klass == "REST":
            for rf in all_RFs:
                if rf not in seen_fams:

                    #call single family loader
                    seqs = seq_loader(datapath, rf, 'fasta_unaligned.txt')
                    seqs_index = seq_to_nt_ids(seqs)
                    fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len)


                    data.append(fixed_seqs)
                    labels.append([rf for _ in range(len(fixed_seqs))])
                    labels_numeral.append([i for _ in range(len(fixed_seqs))])
                    seeds.append(seqs)
            break

        elif klass == "RANDOM":
            if first_fam is None:
                first_fam = 'RF00005'

            seqs_index = generate_based_on_family(first_fam, datapath=datapath)
            fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len)
            data.append(fixed_seqs)
            labels.append([(first_fam + "_RAND") for _ in range(len(fixed_seqs))])
            labels_numeral.append([i for _ in range(len(fixed_seqs))])
            seeds.append(['RAND' for _ in range(len(fixed_seqs))])

        elif klass == "ALL":
            for rf in all_RFs:
                if rf not in seen_fams:

                    #call single family loader
                    seqs = seq_loader(datapath, rf, 'fasta_unaligned.txt')
                    seqs_index = seq_to_nt_ids(seqs)
                    fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len)


                    data.append(fixed_seqs)
                    labels.append([rf for _ in range(len(fixed_seqs))])
                    labels_numeral.append([i for _ in range(len(fixed_seqs))])
                    seeds.append(seqs)
                    i += 1
            break

        else:
            if first_fam is None:
                first_fam = klass
            if klass not in seen_fams:
                seen_fams.add(klass)
                seqs = seq_loader(datapath, klass, 'fasta_unaligned.txt')
                seqs_index = seq_to_nt_ids(seqs)
                fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len)
                data.append(fixed_seqs)
                labels.append([klass for _ in range(len(fixed_seqs))])
                labels_numeral.append([i for _ in range(len(fixed_seqs))])
                seeds.append(seqs)

    seeds = np.concatenate(seeds)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    labels_numeral = np.concatenate(labels_numeral)
    data_df = pd.DataFrame(data)
    labels_df = pd.DataFrame({'RFAM': labels, 'seed': seeds, 'numeral': labels_numeral})
    return data_df, labels_df


def load_single_family(RF, task, datapath="data", max_len=500):
    if task == 'ZP':
        seqs = seq_loader(datapath, RF, 'fasta_unaligned.txt')
        seqs_index = seq_to_nt_ids(seqs)
        fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len)
        return fixed_seqs

    elif task == 'RP':
        seqs = seq_loader(datapath, RF, 'fasta_unaligned.txt')
        seqs_index = seq_to_nt_ids(seqs)
        fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len, random="uniform")
        return fixed_seqs

    elif task == 'NUCSHFLZP':
        seqs_index = shuffle_seqs_in_family(RF, datapath=datapath)
        fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len)
        return fixed_seqs

    elif task == 'NUCSHFLRP':
        seqs_index = shuffle_seqs_in_family(RF, datapath=datapath)
        fixed_seqs = pad_to_fixed_length(seqs_index, max_length=max_len, random="uniform")
        return fixed_seqs

    elif task == 'FMLM1':
        pass

#Takes as input 1) a dictionary of default values - a base h-param grid
#and 2) a dictionary of lists - the variable h-params
#returns a list of dictionaries, corresponding to the complete hyperparameter grid, with the
#variable parameters changed and the default parameters in all other places
#Ex:
#   default_dict = {a: 5, b: "banana", c: int}
#   variable_dict = {a: [4,5], b:["orange","banana"]}
#   output =   [{a: 4, c: "banana", d: int},
#               {a: 4, c: "orange", d: int},
#               {a: 5, c: "banana", d: int},
#               {a: 5, c: "orange", d: int}]
#
def grid_generator(default_dict, variable_dict):
    key, value = zip(*variable_dict.items())
    all_dicts = [dict(zip(key, value)) for value in itertools.product(*value)]
    for key in default_dict.keys():
        if key not in variable_dict.keys():
            for setup in all_dicts:
                setup[key] = default_dict[key]

    return all_dicts

