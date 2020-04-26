from utility import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from sklearn import metrics
import numpy as np
import pdb

def run_epoch (model, dl, loss_fn, optimizer, device, mode="TRAIN"):

    if mode=='TRAIN':
        model.train()
    else:
        model.eval()

    #These metrics are calculated on train and test
    losses = []
    accuracies = []

    #These values are recorded for the AUCs and confusion matrices.
    all_y_scores = []
    all_y_trues = []

    for x, y in dl:
        x = Variable(x).to(device)
        y = Variable(y).to(device)

        model.zero_grad()

        out = model(x).to(device)

        loss = loss_fn(out, y.squeeze().long())
        acc = torch.eq(out.argmax(dim=-1), y.long().flatten()).float().view(-1, 1)
        a = float(acc.mean().detach().cpu().numpy())

        yscores = np.exp(out.detach().cpu().numpy())
        all_y_scores.append(yscores)
        all_y_trues.append(y.detach().cpu().numpy())

        losses.append(float(loss))
        accuracies.append(a)

        #update parameters only if training
        if mode=='TRAIN':
            loss.backward()
            optimizer.step()

    all_y_scores = np.concatenate(all_y_scores)
    all_y_trues = np.concatenate(all_y_trues)
    all_y_trues_onehot = np.zeros((int(all_y_trues.shape[0]), int(all_y_trues[:,0].max()+1)))
    all_y_trues_onehot[np.arange(int(all_y_trues.shape[0])), all_y_trues[:,0].astype("int32")] = 1

    loss_out = np.array(losses).mean()
    acc_out = np.array(accuracies).mean() * 100

    auc_out = metrics.roc_auc_score(y_true=all_y_trues_onehot, y_score=all_y_scores, multi_class='ovo')
    conf_mat = metrics.confusion_matrix(all_y_trues, np.argmax(all_y_scores, axis=-1),
                                        labels=np.arange(all_y_scores.shape[-1], dtype=int))

    return loss_out, acc_out, auc_out, conf_mat


def train (model, tr_dataloader, val_dataloader, model_specs, device = 'cuda:0', foldn = 0):
    epochs = model_specs['epochs']
    wd = model_specs['wd']
    lr = model_specs['lr']
    loss_fn = model_specs['lossfn']
    gr_steps = model_specs['gr_steps']
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)

    # levels_sorted_by_index = sorted([(index, c) for (c, index) in model_specs['levels'].items()])
    # levels = [str(c)[:10] for (ind, c) in levels_sorted_by_index]

    training_loss, training_acc, training_auc = [],[],[]
    validation_loss, validation_acc, validation_auc = [], [], []

    for i in range(epochs):

        tr_loss, tr_acc, tr_auc, tr_conf_mat = run_epoch(model, tr_dataloader, loss_fn, optimizer, device, "TRAIN")

        training_loss.append(tr_loss)
        training_acc.append(tr_acc)
        training_auc.append(tr_auc)

        training_reporter = 'FOLD {} \tTRAIN ['.format(str(foldn).zfill(3)) + ''.join([['#', '.'][int(j > int((i + 1.) * 10 / epochs))] for j in range(10)]) + '] [{}/{}] Avg. acc = {} % | Avg. loss = {}\t'.format(i+1, epochs, round(tr_acc, 3), round(tr_loss, 3))

        print(training_reporter)

        val_loss, val_acc, val_auc, val_conf_mat = run_epoch(model, val_dataloader, loss_fn, optimizer, device, "VALID")

        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        validation_auc.append(val_auc)

        validation_reporter = 'FOLD {} \tVALID ['.format(str(foldn).zfill(3)) + ''.join([['#', '.'][int(j > int((i + 1.) * 10 / epochs))] for j in range(10)]) + '] [{}/{}] Avg. acc = {} % | Avg. loss = {}\t'.format(i+1, epochs, round(val_acc, 3), round(val_loss, 3))

        print(validation_reporter)

    # update model_specs
    model_specs['tr_l'] = np.array(training_loss)
    model_specs['tr_acc'] = np.array(training_acc)
    model_specs['val_l'] = np.array(validation_loss)
    model_specs['val_acc'] = np.array(validation_acc)
    model_specs['tr_auc'] = np.array(training_auc)
    model_specs['val_auc'] = np.array(validation_auc)
    model_specs['tr_conf_mat'] = tr_conf_mat
    model_specs['val_conf_mat'] = val_conf_mat
