import pandas as pd
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn import manifold
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
perplexities = [35,100, 500, 1000, 10000]

# prt_data_tpm_embedded = pd.read_csv('TSNE/prt_data_tpm_embedded_p_15.txt')
# trsc_data_tpm_embedded = pd.read_csv('TSNE/trsc_data_tpm_embedded_p_15.txt')

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
def apply_aesthetics():
        # Overriding styles for current script
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.bottom'] = False
        plt.rcParams['axes.spines.left'] = False

target = trsc_data_tpm.index.isin(prt_data_tpm.index)
target = [x for x in map(lambda x: ['PRT','N-CDING'][x], np.array(target, dtype = int) )] 
path = 'TSNE/BY_GENE'
# reformat string 
assert_mkdir(path)
print ('removing non-expressed genes ...')
target = np.array(target)[trsc_data_tpm.sum(axis  = 1) != 0]
trsc_data_tpm = trsc_data_tpm[trsc_data_tpm.sum(axis = 1) != 0] # remove non expressed genes

for perplexity in perplexities:
        # run tsne on protein coding 
        print ('[prgrm] perp =  {} TPM TRSC BY GENE: '.format(perplexity), 'fitting tsne ...')  
        trsc_data_embedded = manifold.TSNE(n_components = 2, n_iter = 2000, verbose = 2 , perplexity = perplexity, init = 'pca').fit_transform(trsc_data_tpm.loc[:,samples].values)
        trsc_data_embedded_DF = pd.DataFrame(trsc_data_embedded, columns = ['TSNE1','TSNE2'], index = trsc_data_tpm.index)
        outfile = os.path.join(path, 'trsc_data_gene_embedding_p_{}.txt'.format(perplexity))
        print ('[prgm] saving file ... {}'.format(outfile))
        trsc_data_embedded_DF.to_csv(outfile)
        #trsc_data_embedded_DF = pd.read_csv('TSNE/BY_GENE/trsc_data_gene_embedding_p_15.txt', index_col = 0)        
        # for perplexity in perplexities:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10 ))
        plt.style.use('custom')
        apply_aesthetics()
        
        print ('plotting ...') 

        for num, l in enumerate(np.unique(target)[::-1]):
                marker = 'o'
                if num > 9: marker = '^'
                trsc_tsne = trsc_data_embedded_DF[target == l]
                ax.scatter(trsc_tsne['TSNE1'], trsc_tsne['TSNE2'], label = '{} : [n = {}]'.format(l, sum(target == l)), edgecolor = 'w', marker = marker)

        ax.title.set_text('TSNE TPM ON TRSCPTOME BY GENE: PERPLEXITY = {}'.format(perplexity))

        ax.legend()

        plt.tight_layout()    
        path = 'public_html/fig/TSNE/BY_GENE'
        assert_mkdir(path)
        plt.savefig(os.path.join(path, 'TSNE_56K_20K_p{}.pdf'.format(perplexity)))
        plt.savefig(os.path.join(path, 'TSNE_56K_20K_p{}.png'.format(perplexity)))

