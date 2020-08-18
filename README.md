# Elucidating the Automatically Detected Features Used By Deep Neural Networks for RFAM family classification
### Tom MacDougall, Léonard Sauvé, François Major, Sébastien Lemieux
Work presented as a poster and talk at ISMB 2020

Command for full environment setup using conda:

    conda create -y -n RNAClassification python=3.7 scipy pandas matplotlib seaborn scikit-learn pytorch torchvision cudatoolkit=10.1 -c pytorch

This is a repository of the code and data used in our ISMB 2020
abstract submission and as our IFT6292 final project

##### The scripts/ directory contains all scripts used in this project

TrainDNN_multiclass.py in the main runnable script that, despite its name, 
performs both binary and multiclass classification

train_multiclass.py implements the training loop and training structure for the method

utilities.py implements various data-loading and data-related functions

load_exp_data.py can re-generate figures from past experiments based on saved data

##### The data/ directory contains the classification data, and the structure is very precise. 

A data directory contains many sub-directories, one for each class, named according
to what class it is. Each sub-directory contains a file called "fasta_unaligned.txt"
which contains the sequences of that class

The 7405 sequences in 24 Rfam families that we used are found in the 
data.zip archive in this repo
