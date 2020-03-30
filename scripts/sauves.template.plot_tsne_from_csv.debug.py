import pandas as pd
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler 
np.random.seed(111)
print('loading files...')
labels_file = pd.read_csv('/u/sauves/leucegene/data/patients/patients.20190922.txt', sep = '\t')
labels_file['illumina_sequencer'] = [['Hi-seq', 'Nova-seq'][int(dsp_ext in ['EXT003', 'EXT008'])] for dsp_ext in labels_file['Genomic_Project_ID']]
labels_file['dx_FAB_alt'] = [str(x)[:6] for x in labels_file['dx_FAB']] 
sample_fnames = pd.read_csv('~/paper_strandedness/data/COUNT/star.header.merged.2019-10-25-13:30:25.csv')
sample_fnames['pname'] = [x for x in map(lambda x: x.split('/')[7], np.array(sample_fnames.sample_fname))]
sample_fnames['genomic_project_ID'] = [x for x in map(lambda x: x.split('/')[6], np.array(sample_fnames.sample_fname))]
sample_fnames['fullname'] = [d + '/' +  p for (d,p) in zip(sample_fnames.genomic_project_ID, sample_fnames.pname)]
samples = labels_file.merge(sample_fnames, on='pname').fullname
prt_data_tpm = pd.read_csv('/u/sauves/paper_beats/data/COUNT/star.count.tpm.prt_coding.714s.txt', index_col = 0)
trsc_data_tpm = pd.read_csv('/u/sauves/paper_beats/data/COUNT/star.count.tpm.all_transcripts.714s.txt', index_col = 0)

prt_data_tpm = prt_data_tpm.loc[:,samples]
trsc_data_tpm = trsc_data_tpm.loc[:,samples]
labels = labels_file.merge(sample_fnames, on = 'pname')
perplexities = np.arange(5,700,5) 

prt_data_tpm_embedded = pd.read_csv('TSNE/prt_data_tpm_embedded_p_15.txt')
trsc_data_tpm_embedded = pd.read_csv('TSNE/trsc_data_tpm_embedded_p_15.txt')

def apply_aesthetics():
        # Overriding styles for current script
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.bottom'] = False
        plt.rcParams['axes.spines.left'] = False

for idx, group in enumerate(labels.columns):
# for perplexity in perplexities:
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 10 * 2))
    labels[group][labels[group] != labels[group]] = 'not avail' # remove nans
    labels[group] = np.array(labels[group], dtype = str) # convert to factors
    plt.style.use('custom')
    apply_aesthetics()
    if (len(np.unique(labels[group])) < 22) and (len(np.unique(labels[group])) > 1):
            print ('[{}] plotting [{}]...'.format(idx, group)) 
            labels[group][labels[group] == '-'] = 'not avail' # uniform nans
            
            ax.ravel()[0].scatter(prt_data_tpm_embedded[labels[group] == 'not avail']['TSNE1'], prt_data_tpm_embedded[labels[group] == 'not avail']['TSNE2'], label = 'not avail : [n = {}]'.format(sum(labels[group] == 'not avail')), edgecolor = 'k', facecolor = 'grey', alpha = 0.5)
            ax.ravel()[1].scatter(trsc_data_tpm_embedded[labels[group] == 'not avail']['TSNE1'], trsc_data_tpm_embedded[labels[group] == 'not avail']['TSNE2'], label = 'not avail', edgecolor='k', facecolor = 'grey', alpha = 0.5)
            for num, l in enumerate(list(set(np.unique(labels[group])) - set(['not avail']))):
                marker = 'o'
                if num > 9: marker = '^'
                ax.ravel()[0].scatter(prt_data_tpm_embedded[labels[group] == l]['TSNE1'], prt_data_tpm_embedded[labels[group] == l]['TSNE2'], label = '{} : [n = {}]'.format(l, sum(labels[group] == l)), edgecolor = 'w', marker = marker)
                ax.ravel()[1].scatter(trsc_data_tpm_embedded[labels[group] == l]['TSNE1'], trsc_data_tpm_embedded[labels[group] == l]['TSNE2'], label = l, edgecolor='w', marker = marker)
                
                ax.ravel()[0].title.set_text('TSNE TPM 20K TRSCPTOME: {}'.format(group))
                ax.ravel()[1].title.set_text('TSNE TPM 56K TRSCPTOME: {}'.format(group))
                
                ax.ravel()[0].legend()
            
            plt.tight_layout()    
            group = group.replace(' ', '_').replace('/','_').replace('-','_')
            plt.savefig('/u/sauves/public_html/fig/TSNE/BY_GROUP/[{}]_{}_TSNE_56K_20K_p15.pdf'.format(idx,group))
            plt.savefig('/u/sauves/public_html/fig/TSNE/BY_GROUP/[{}]_{}_TSNE_56K_20K_p15.png'.format(idx,group))

    else : print('[{}] too many feature types, not groupable by [{}]. Label data is probably not categorical'.format(idx, group))
