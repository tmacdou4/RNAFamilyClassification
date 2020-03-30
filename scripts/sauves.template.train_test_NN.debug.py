import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader 
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import numpy as np
import pdb
import argparse
import os
import multiprocessing as mp
from collections import Counter
from torch.autograd import Variable
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-seed', dest = 'SEED', default= 1, type = int, help = 'random seed')
parser.add_argument('-epochs',dest = 'EPOCHS', default = 50, type = int, help = 'nb of max epochs')
parser.add_argument('-l', dest = 'LABEL', default = None, type = str, help = 'name of label to train on') 
parser.add_argument('-t', dest = 'THREADING', default = 'gpu', type = str, help = 'thjreading specs: [gpu, cpu-X]')
parser.add_argument('-n', dest = 'N_THREADS' , type = int, help = 'number of parallel processes')
parser.add_argument('-wd', dest = 'WEIGHT_DECAY', type = float, default = 0.2, help = 'L2 parametrization [0:no regularization]')
parser.add_argument('-logistic', dest = 'LOGISTIC', action = 'store_true', help = 'switch on the logistic regression fit on data')
parser.add_argument('-hid1n', dest = 'HID1N', default=8, type = int, help = 'number of nodes in 1st layer')
parser.add_argument('-hid2n', dest = 'HID2N', default=0, type = int, help = 'number of nodes in 2st hidden layer (if any)')
parser.add_argument('-xval', dest = 'XVAL', default =10, type = int, help= 'number of folds for crossvalidation')
parser.add_argument('-cuda', dest = 'CUDA', default = 1, type = int, help= 'device ID of gpu to use')
parser.add_argument('-input_genes', dest = 'INPUT_GENES', default = 'PRT', type = str, help = 'transcript type used for training, chose among [PROTEIN CODING ("PRT") , NON-CODING("NCOD") , ALL("ALL")]' )
args = parser.parse_args()


class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data['data'][idx]), torch.LongTensor(np.array(self.data['labels'][idx]))
    def __len__(self):
        return self.data['data'].shape[0]

class ProportionalDataPicker(Dataset):
        def __init__(self, data):
            self.data = data
            self.size = self.data['data'].shape[0]
            self.class_counter = Counter(self.data['labels'])
            self.class_densities = dict([(classe, float(count) / self.size) for (classe, count) in self.class_counter.items()])
            self.idx_probability = [self.class_densities[label] / self.class_counter[label] for label in self.data['labels']]
        def __getitem__(self, idx):
            # first pick an index at 'random' following class densities    
            idx = np.random.choice(np.arange(self.size), p = self.idx_probability)
            # then return x,y tuple for this index
            return torch.FloatTensor(self.data['data'][idx]), torch.LongTensor(np.array(self.data['labels'][idx])) 
        def __len__(self):
            return self.size

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
    
class Model2H(nn.Module):
        def __init__(self,model_specs):
                super(Model2H, self).__init__()
                self.in_h1 = nn.Linear(model_specs['input_size'], model_specs['HID1N'])
                self.h1_nl = nn.Hardtanh()
                self.h1_h2 = nn.Linear(model_specs['HID1N'], model_specs['HID2N'])
                self.h2_nl = nn.Hardtanh()
                self.h2_out = nn.Linear(model_specs['HID2N'], model_specs['output_size'])
                self.out_nl = nn.Softmax(dim=-1)
        def forward(self, x):
            out = self.in_h1(x)
            out = self.h1_nl(out)
            out = self.h1_h2(out)
            out = self.h2_nl(out)
            out = self.h2_out(out)
            out = self.out_nl(out)
            return out                    

class Model1H(nn.Module):
        def __init__(self,model_specs):
                super(Model1H, self).__init__()
                self.drop_in = nn.Dropout(model_specs['dropout1'])
                self.in_h1 = nn.Linear(model_specs['input_size'], model_specs['HID1N'])
                self.h1_nl = nn.Hardtanh()
                self.h1_out = nn.Linear(model_specs['HID1N'], model_specs['output_size'])
                self.out_nl = nn.Softmax(dim=-1)
        def forward(self, x):
            out = self.drop_in(x)
            out = self.in_h1(out)
            out = self.h1_nl(out)
            out = self.h1_out(out)
            out = self.out_nl(out)
            return out 
class LogisticRegression(nn.Module):
        def __init__(self, model_specs):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(model_specs['input_size'], model_specs['output_size'])
        def forward(self, x):
            out = self.linear(x)
            return out

def train (model, dataloader, vld_dl, model_specs,device = None, fold = None):
    if model_specs['model'] == 'LOGISTIC':
        epochs = model_specs['epochs']
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, weight_decay = model_specs['wd'])
        for e in range(epochs):
            for x, y in dataloader:
                optimizer.zero_grad()
                out = model(x)
                loss_val = loss_fn(out, y)
                loss_val.backward()
                optimizer.step()
    else : 
        epochs = model_specs['epochs']
        wd = model_specs['wd']
        lr = model_specs['lr']
        loss = torch.nn.NLLLoss() if model_specs['lossfn'] == 'NLL' else torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
        levels_sorted_by_index = sorted([(index, c) for (c, index) in model_specs['levels'].items()])
        levels = [str(c)[:10] for (ind, c) in levels_sorted_by_index]
        frame_nb = 0
        for i  in xrange(epochs):
            n = 0
            l = 0
            a = 0
            m = 0 
            auc = 0
            TP, FN, FP, TN = 0,0,0,0
            CM = np.zeros((model_specs['output_size'], model_specs['output_size']))
            # fig = plt.figure(figsize= (10,10))
            # fig.add_subplot(111, projection = '3d')
            for x,y in dataloader:
                x = Variable(x).to(device)
                y = Variable(y).to(device)
                # pdb.set_trace()
                out = model(x)
                out_cpy = out.detach().cpu().numpy()
                out_df = pd.DataFrame(out_cpy, columns = model_specs['levels_list'])
                out_df['true_label'] = [model_specs['levels_list'][c] for c in y.detach().cpu().numpy()]
                out_df.to_csv(os.path.join(TRAINING_outpath, str(fold).zfill(3), str(frame_nb).zfill(4)), index = False)
                frame_nb += 1
                # ax.scatter(out_cpy[:,0], out_cpy[:,1], out_cpy[:,2], marker = out_cpy.argmax(axis = 1))
                # plt.savefig('TRAINING/{}/{}.png'.format(model_specs['label'], str(n).zfill(3)), dpi = 100) 
                loss_val = loss(out, y).view(-1, 1).expand_as(out)
                acc = torch.eq(out.argmax(dim = -1), y).float().view(-1,1)
                if model_specs['output_size'] <= 2:              
                    tn, fp, fn, tp = metrics.confusion_matrix(y, out.argmax(dim=-1).detach().cpu().numpy(), labels = np.arange(2, dtype=int)).ravel()
                    TN += tn
                    FP += fp
                    FN += fn
                    TP += tp
                    auc += metrics.roc_auc_score(y_true = y, y_score = out[:,1].detach().cpu().numpy())
                else:
                    CM += metrics.confusion_matrix(y.detach().cpu().numpy(), out.argmax(dim=-1).detach().cpu().numpy(), labels = np.arange(model_specs['output_size']))
                # m += metrics.matthews_corrcoef(y, out.argmax(dim = -1))
                # pdb.set_trace()
                n += 1
                a += acc.mean().detach().cpu().numpy()
                l += float(loss_val.mean())
                optimizer.zero_grad()
                loss_val.mean().backward()
                optimizer.step()
            
            output_fpath = os.path.join('NN',args.LABEL,args.INPUT_GENES, 'PROGRESS','FOLD{}'.format(str(fold).zfill(3)))
            if (TN + FN + FP + TP) != 0:
                AUC_score = round(auc / n,3) 
                training_reporter  = 'FOLD {}\tTRAIN ['.format(str(fold).zfill(3)) + ''.join([['#','.'][int(j > int((i + 1.) * 10/epochs))] for j in range(10) ]) + '] [{}/{}] {} % AUC {} [TN:{}, FP:{}, FN:{}, TP:{}]'.format(i+1, epochs, round(a / n, 4) * 100, AUC_score, TN, FP, FN, TP)    
                os.system('echo {} > {}'.format(training_reporter,output_fpath))
            else :
                training_reporter = 'FOLD {}\tTRAIN ['.format(str(fold).zfill(3)) + ''.join([['#','.'][int(j > int((i + 1.) * 10/epochs))] for j in range(10) ]) + '] [{}/{}] {} % '.format(i+1, epochs, round(a / n, 4) * 100)
                os.system('echo {} > {}'.format(training_reporter,output_fpath))
                pd.DataFrame(CM.T, dtype = int, index = levels, columns = levels).to_csv("{}CM".format(output_fpath), sep = "\t")
            # print ("MODEL{} TRAIN Loss: {}".format(str(i).zfill(2), l / n))
            # print ("MODEL{} TRAIN Accuracy: {}".format(str(i).zfill(2), a / n))
            # print ("MODEL{} TRAIN MCC: {}".format(str(i).zfill(2), m/n)) 
            # print ("MODEL{} TRAIN AUC: {}".format(str(i).zfill(2), auc/n)) 
            
            MODEL_PATH = os.path.join('NN', args.LABEL,args.INPUT_GENES, 'MODELS')
            if (i+1) == args.EPOCHS: torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'FOLD{}_MODEL{}'.format(str(fold).zfill(3), str(i).zfill(2))))
            
            for x, y in vld_dl:
                x = Variable(x).to(device)
                y = Variable(y).to(device)
                out = model(x)
                acc = torch.eq(out.argmax(dim = -1), y).float().view(-1,1)
                if model_specs['output_size'] == 2: 
                    AUC = metrics.roc_auc_score(y_true = np.array(y), y_score = out[:,1].detach().cpu().numpy())
                # print "EPOCH {} TRAIN ACC: {} % VALID: {} % AUC: {}".format(i+1, round(a/n, 4) * 100, round(acc.mean(), 4) * 100, round(AUC,3))
                # pdb.set_trace()
    
def test (model,test_dataloader, device = None):
        for x, y in test_dataloader:
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            out = model(x)
            acc = torch.eq(out.argmax(dim = -1), y).float().view(-1,1)
            acc = np.array(acc.detach().cpu().numpy().flatten(), dtype = int)
        return acc, out
            

def train_test_cross_val(data, labels,model_specs,xval = 10,  device= None):
        # init results arrays
        ACC_results = []
        AUC_results = []
        ytrue_array = []
        yscores_array = []
        # store some static values 
        nsamples = model_specs['nsamples']
        test_size = model_specs['test_size'] 
        # begin crossvalidation
        for foldn in range(xval):
            # prepare data splitting    
            samples = labels.index[foldn * test_size: min((foldn + 1) * test_size, nsamples)] 
            TRAIN_X = data.loc[:, set(labels.index) - set(samples)]
            TRAIN_Y = labels.loc[set(labels.index) - set(samples),:]
            TEST_X = data.loc[:,samples]
            TEST_Y = labels.loc[samples,:]
                
            # prepare data loaders 
            if model_specs['loader'] == 'balanced' : dataset  = BalancedDataPicker({'data': np.array(TRAIN_X.T),'labels':np.array(TRAIN_Y.numeral)})
            elif model_specs['loader'] == 'proportional' : dataset  = ProportionalDataPicker({'data': np.array(TRAIN_X.T),'labels':np.array(TRAIN_Y.numeral)})
            else : dataset  = Dataset({'data': np.array(TRAIN_X.T),'labels':np.array(TRAIN_Y.numeral)})
            dl = DataLoader(dataset, batch_size = model_specs['batch_size'])
            test_dataset = Dataset({'data': np.array(TEST_X.T), 'labels':np.array(TEST_Y.numeral)})
            test_dl = DataLoader(test_dataset, batch_size = len(TEST_Y.index))    
            # init model 
            if model_specs['n_hid_lyrs'] == 1 : model = Model1H(model_specs).cuda(device)
            elif model_specs['n_hid_lyrs'] == 2: model = Model2H(model_specs).cuda(device) 
            else : model = LogisticRegression(model_specs)
            # save checkpoint
            init_weights = dict([(key, torch.FloatTensor(val.detach().cpu().numpy())) for (key,val) in model.state_dict().items()])
            
            # train model 
            train(model, dl, test_dl, model_specs, device = device, fold=foldn) 
            # save checkpoint 
            trained_weights = dict([(key, torch.FloatTensor(val.detach().cpu().numpy())) for (key,val) in model.state_dict().items()])
            # has_the_model_learned_anything = [np.sum(init_weights[k] != trained_weights[k]) for k in trained_weights.keys()]
            # begin tests
            output_fpath = os.path.join(model_specs['model'],args.LABEL, args.INPUT_GENES, 'PROGRESS','FOLD{}'.format(str(foldn).zfill(3)))
            acc, out = test(model, test_dl, device)
            AUC = 0 
            ACC_results.append(acc)            
            if (TEST_Y['numeral'].max() < 2) & (len(np.unique(TEST_Y['numeral'])) > 1) :
                    tn, fp, fn, tp = metrics.confusion_matrix(np.array(TEST_Y['numeral']), out.argmax(dim=-1).detach().cpu().numpy()).ravel()
                    ytrue =  np.array(TEST_Y['numeral'])
                    yscores =  out[:,1].detach().cpu().numpy()
                    AUC = metrics.roc_auc_score(y_true = ytrue, y_score = yscores)
                    ytrue_array.append(ytrue)
                    yscores_array.append(yscores)
                    AUC_results.append(AUC)
                    os.system('echo "FOLD {} TEST\t{} %\tAUC: {} [TN:{}, FP:{}, FN:{}, TP:{}]" > {}TEST'.format(str(foldn).zfill(3),round(acc.mean(),3) * 100,round(AUC,3),tn, fp, fn, tp, output_fpath))
            else :
                CM = metrics.confusion_matrix(np.array(TEST_Y['numeral']), out.argmax(dim=-1).detach().cpu().numpy())
                os.system('echo "FOLD {} TEST\t{} %\t Confusion Matrix :\n {}" > {}TEST'.format(str(foldn).zfill(3),round(acc.mean(),3) * 100,CM.T, output_fpath)) 
        if len(AUC_results) > 0:   
            output_data = {'auc': np.array(AUC_results), 'ytrue': np.array(ytrue_array), 'yscores': np.array(yscores_array) ,'model_specs':model_specs}
        else:
                output_data = {'auc': None, 'ytrue' : np.array(ytrue_array), 'ypreds': np.array(yscores_array), 'model_specs':model_specs}
        return np.array(ACC_results), output_data

# some utility functions
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

def echo_write(R, F, P):
    """ 
    function that writes the string R in a file F on path P
    """
    os.system("echo '{}' > {}".format(R, os.path.join(P,F)))

def cat_write(R, O, P):
    """
    function that writes the content R of Directory P into file F in deirectory P
    """
    if not os.path.exists(os.path.join(P, O)):
            os.system("cat {} > {}".format(os.path.join(P, R), os.path.join(P, O)))

def MEAN_STR(A, zf = 3):
    """
    Computes mean of np.array of floats A [0,1], then converts to string with zf zero fill.
    """
    return str(int(A.mean() * 100)).zfill(zf)

def train_test_logistic_regression():
        print ('Training logistic regression model, searching for best parameter L2 ...')
        worst_model = 1
        best_model = 0
        model_data = dict([('best', dict()),('worst', dict())])
        
        # perform small gridsearch for best L2 parameter 
        for l2_reg in np.arange(0,10, dtype = float) / 10:
            
            # Logistic model
            model_specs = {'xval' : args.XVAL, 
                            'nsamples': labels.shape[0],
                            'test_size': int(float(labels.shape[0]) / args.XVAL),
                            'model' : 'NN',
                            'n_hid_lyrs': 0,
                            'loader': 'normal',
                            'output_size': labels['numeral'].max() + 1,
                            'lr' : 1e-4,
                            'dropout1': 0,
                            'HID1N' : 0,
                            'HID2N' : 0,
                            'lossfn' : 'CrossEntropy', 
                            'levels' : numeric_labels,
                            'model': 'LOGISTIC', 
                            'epochs' : 10, 
                            'batch_size' : len(labels.index) - int(float(len(labels.index)) / xval), # all samples
                            'input_size': data.shape[0], # all input features 
                            'wd' : l2_reg, # only parameter to optimize
                            }
            # report model specs in a resfile
            echo_write(R = str(model_specs), P = LOGISTIC_res_path, F = 'model_specs.txt')
            
            # perform LOGISTIC REGRESSION model training-testing in cross val 
            accuracy_res, auc_data_log = train_test_cross_val(data, labels, model_specs, device = 'cpu', xval = xval)

            # report accuracy in a resfile
            r = "ACCURACY BY {} FOLD CROSSVAL AFTER {} EPOCHS OF TRAININING ON {} SAMPLES\n {} % | AUC: {}".format(xval, model_specs['epochs'], len(labels.index), MEAN_STR(accuracy_res, zf = 0), MEAN_STR(auc_data_log['auc']))
            echo_write(R = r, P = LOGISTIC_res_path, F = 'TEST_RESULT_LOGISTIC')
            
            # update worst and best model
            if auc_data_log['auc'].mean() > best_model:
                    model_data['best'] =  auc_data_log
                    best_model = auc_data_log['auc'].mean()
            elif auc_data_log['auc'].mean() < worst_model: 
                    model_data['worst'] = auc_data_log
                    worst_model = auc_data_log['auc'].mean()
            print ('L2 (weight decay) = ',l2_reg,' : ', round(auc_data_log['auc'].mean(),4) * 100)

        # concatenate report results 
        fname = 'AUC_{}_TEST_RESULTS_LOGISTIC_L2_OPTIM.txt'.format(MEAN_STR(model_data['best']['auc']))
        cat_write(R = '*', P = LOGISTIC_progress_path, O = fname)
        
        # plot roc-auc curves 
        # plot_roc_auc_curve(model_data, fname) 
        
        # set a scores reporter for ROC-curve tracing 
        r = pd.DataFrame(dict([['scores', model_data['best']['yscores'].flatten()], ['true', model_data['best']['ytrue'].flatten()]]))
        r.to_csv(os.path.join(LOGISTIC_res_path, '{}_auc_scores.txt'.format(MEAN_STR(model_data['best']['auc']))))



def plot_roc_auc_curve(models1_2, fname, xval = 10):
    fig, ax =  plt.subplots(figsize=(20, 20))
    for id, model_data in enumerate(models1_2):
        for foldn in xrange(xval):
            # fix threshold
            TPR = dict()
            FPR = dict()
            ypred_foldn = np.array(model_data['yscores'][foldn])
            ytrue_foldn = model_data['ytrue'][foldn]
            for i in range(2):
                    TPR[i],FPR[i], _ = metrics.roc_curve (ytrue_foldn, ypred_foldn)
            
            AUC = metrics.roc_auc_score(y_true = model_data['ytrue'][foldn] , y_score = model_data['yscores'][foldn] )
            color = ['darkblue','magenta'][id]
            label = '{} L2: {}'.format(model_data['model_specs']['model'], model_data['model_specs']['wd'])
            label = label + '- FOLD {} - AUC: {}'.format(str(foldn).zfill(2) , round(AUC, 2))
            ax.plot(TPR[0],FPR[0], label = label, lw = 4, alpha = 0.5, c = color)
        
    plt.legend()
    plt.title('ROC-curves of the classification models to predict {}'.format(args.LABEL))
    plt.xlabel('True Positives Rate (TPR)')
    plt.ylabel('False Positives Rate (FPR)')
    plt.savefig('{}.pdf'.format(fname))


# data preprocessing 
print('[{}]\nloading files...'.format(args.LABEL))
data = pd.read_csv("/u/sauves/paper_beats/data/COUNT/star.count.tpm.all_transcripts.714s.txt", index_col = 0)

labels_file = pd.read_csv('/u/sauves/leucegene/data/patients/patients.20190922.txt', sep = '\t')
labels_file['illumina_sequencer'] = [['Hi-seq', 'Nova-seq'][int(dsp_ext in ['EXT003', 'EXT008'])] for dsp_ext in labels_file['Genomic_Project_ID']]
labels_file.columns = [col.replace(' ','_').replace('/','_').replace('-','_') for col in np.array(labels_file.columns)]
sample_fnames = pd.read_csv('~/paper_strandedness/data/COUNT/star.header.merged.2019-10-25-13:30:25.csv')
sample_fnames['pname'] = [x for x in map(lambda x: x.split('/')[7], np.array(sample_fnames.sample_fname))]
sample_fnames['genomic_project_ID'] = [x for x in map(lambda x: x.split('/')[6], np.array(sample_fnames.sample_fname))]
sample_fnames['fullname'] = [d + '/' +  p for (d,p) in zip(sample_fnames.genomic_project_ID, sample_fnames.pname)]
pdb.set_trace()
labels = labels_file.merge(sample_fnames, on='pname')
labels = pd.DataFrame(labels[[args.LABEL, 'fullname']])
labels = labels[labels[args.LABEL] == labels[args.LABEL]] # remove nans in label file
labels = labels[labels[args.LABEL] != '-']
data_samples = labels.merge(pd.DataFrame(data.columns, columns = ['fullname']), on = 'fullname', how = 'inner').fullname
labels.index = labels.fullname
labels = labels.loc[data_samples,:]

prt_20K_genes = pd.read_csv('/u/sauves/paper_beats/data/COUNT/star.count.tpm.prt_coding.714s.txt', index_col = 0).index
trsc_56K_genes = pd.read_csv('/u/sauves/paper_beats/data/COUNT/star.count.tpm.all_transcripts.714s.txt', index_col = 0).index
ncod_35K_genes = trsc_56K_genes.difference(prt_20K_genes)

input_genes = {'PRT': prt_20K_genes , 'NCOD': ncod_35K_genes ,'ALL': trsc_56K_genes}[args.INPUT_GENES] 
data = data.loc[input_genes, data_samples]
data =  data[data.sum(axis = 1) != 0] # remove non-expressed genes

np.random.seed(args.SEED)
torch.manual_seed(args.SEED) # set seed for model initialization 
numeric_labels = dict(zip(np.unique(labels[args.LABEL]), np.arange(len(np.unique(labels[args.LABEL])))))
labels['numeral'] = [numeric_labels[l] for l in labels[args.LABEL]]
rnd_idxs = np.arange(labels.shape[0])
np.random.shuffle(rnd_idxs)
labels = labels.iloc[rnd_idxs]
data = data.iloc[:,rnd_idxs]

# prepare_outfile_paths
NN_label_path = os.path.join('NN', args.LABEL, args.INPUT_GENES)
NN_progress_path = os.path.join(NN_label_path, 'PROGRESS')
NN_models_path = os.path.join(NN_label_path, 'MODELS')
NN_tr_logs_path = os.path.join(NN_label_path, 'TRAINING_LOGS')
NN_res_path = os.path.join(NN_label_path, 'RES')
 
LOGISTIC_label_path = os.path.join('LOGISTIC', args.LABEL, args.INPUT_GENES)
LOGISTIC_progress_path = os.path.join(LOGISTIC_label_path,'PROGRESS')
LOGISTIC_models_path = os.path.join(LOGISTIC_label_path, 'MODELS')
LOGISTIC_tr_logs_path = os.path.join(LOGISTIC_label_path, 'TRAINING_LOGS')
LOGISTIC_res_path = os.path.join(LOGISTIC_label_path, 'RES')   #

assert_mkdir(NN_label_path)
assert_mkdir(NN_progress_path)
assert_mkdir(NN_models_path)
assert_mkdir(NN_tr_logs_path)
assert_mkdir(NN_res_path)

assert_mkdir(LOGISTIC_label_path)
assert_mkdir(LOGISTIC_progress_path)
assert_mkdir(LOGISTIC_models_path)
assert_mkdir(LOGISTIC_tr_logs_path)
assert_mkdir(LOGISTIC_res_path)

TRAINING_outpath = os.path.join('TRAINING', args.LABEL, args.INPUT_GENES)
for foldn in xrange(args.XVAL):
    assert_mkdir(os.path.join(TRAINING_outpath, str(foldn).zfill(3)))

# begin training and testing with predictors
xval = args.XVAL
if args.LOGISTIC: train_test_logistic_regression()
train_size = len(labels.index) - int(float(len(labels.index)) / xval)
# MLP model
model_specs = {
        'xval' : args.XVAL,
        'label': args.LABEL,
        'input_genes': args.INPUT_GENES,
        'nsamples': labels.shape[0],
        'test_size': int(float(labels.shape[0]) / args.XVAL), 
        'model' : 'NN',
        'n_hid_lyrs': [2,1][int(args.HID2N == 0)],
        'loader': 'balanced',
        'input_size' : data.shape[0],
        'output_size': labels['numeral'].max() + 1,
        'batch_size' : 64 , #train_size / 10,
        'wd' : args.WEIGHT_DECAY,
        'lr' : 1e-4,
        'dropout1': 0,
        'HID1N' : args.HID1N,
        'HID2N' : args.HID2N,
        'lossfn' : 'CrossEntropy', 
        'epochs' : args.EPOCHS,
        'levels' : numeric_labels,
        'levels_list': [item for (c, item) in sorted([(c, item) for (item, c) in numeric_labels.items()])]}
os.system('echo "{}" > NN/{}/{}/PROGRESS/MODEL_SPECS'.format(str(model_specs),args.LABEL, args.INPUT_GENES))       
print('trainig model...')
res, auc_data = train_test_cross_val(data, labels, model_specs, device = 'cuda:{}'.format(args.CUDA), xval = xval)
AUC_SCORES = pd.DataFrame(dict([['scores', auc_data['yscores'].flatten()], ['true', auc_data['ytrue'].flatten()]]))
AUC_SCORES.to_csv(os.path.join(NN_res_path, '{}_auc_scores.txt'.format(str(int(auc_data['auc'].mean() * 100)).zfill(3))))

if model_specs['output_size'] <= 2 :
        auc_mean = auc_data["auc"].mean()
        os.system('echo "ACCURACY BY {} FOLD CROSSVAL AFTER {} EPOCHS OF TRAININING ON {} SAMPLES\n {} % | AUC: {}"  > NN/{}/{}/PROGRESS/TEST_RESULT'.format(xval, args.EPOCHS, len(labels.index),round(res.mean(), 4) * 100, auc_mean, args.LABEL, args.INPUT_GENES))
        fname = 'NN/{}/{}/RES/AUC_{}_TEST_RESULTS'.format(args.LABEL, args.INPUT_GENES, str(int(auc_data['auc'].mean() * 100)).zfill(3) )
        os.system('cat NN/{}/{}/PROGRESS/* > {}.txt'.format(args.LABEL,args.INPUT_GENES, fname) ) 
else:
        os.system('echo "ACCURACY BY {} FOLD CROSSVAL AFTER {} EPOCHS OF TRAININING ON {} SAMPLES\n {} %"  > NN/{}/{}/PROGRESS/TEST_RESULT'.format(xval, args.EPOCHS, len(labels.index),round(res.mean(), 4) * 100, args.LABEL, args.INPUT_GENES))
        fname = 'NN/{}/{}/RES/ACC_{}_TEST_RESULTS'.format(args.LABEL, args.INPUT_GENES, str(int(res.mean() * 10000)).zfill(5))
        os.system('cat NN/{}/{}/PROGRESS/* > {}.txt'.format(args.LABEL, args.INPUT_GENES, fname) ) 
    
   
    


