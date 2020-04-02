from utility import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn import metrics
import pdb

def train (model, dataloader, model_specs, device = 'cuda:0', foldn = 0):
    epochs = model_specs['epochs']
    wd = model_specs['wd']
    lr = model_specs['lr']
    loss = model_specs['lossfn']
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    # levels_sorted_by_index = sorted([(index, c) for (c, index) in model_specs['levels'].items()])
    # levels = [str(c)[:10] for (ind, c) in levels_sorted_by_index]
    frame_nb = 0
    for i  in range(epochs):
        n = 0
        l = 0
        a = 0
        m = 0 
        auc = 0
        TP, FN, FP, TN = 0,0,0,0
        mcc = 0
        # CM = np.zeros((model_specs['output_size'], model_specs['output_size']))
        for x,y in dataloader:
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            out = model(x).to(device)
            n += 1
            loss_val = loss(out[:,1], y).view(-1, 1).expand_as(out)
            # loss, accuracy, and other performance metrics
            acc =torch.eq(out.argmax(dim = -1), y.long().flatten()).float().view(-1,1)
            a += float(acc.mean().detach().cpu().numpy())
            l += float(loss_val.mean())
            tn, fp, fn, tp = metrics.confusion_matrix(y.detach().cpu().numpy(), out.argmax(dim=-1).detach().cpu().numpy(), labels = np.arange(2, dtype=int)).ravel() 
            yscores =  out[:,1].detach().cpu().numpy()
            AUC = metrics.roc_auc_score(y_true = y.detach().cpu().numpy(), y_score = yscores)
            # gradient steps 
            optimizer.zero_grad()
            loss_val.mean().backward()
            optimizer.step()
            #  print ('gradient step [ {} / {} ]'.format(n, model_specs['gr_steps']))
            training_reporter = 'FOLD {} STEP[{}/{}]\tTRAIN ['.format(str(foldn).zfill(3), n ,model_specs['gr_steps'] ) + ''.join([['#','.'][int(j > int((i + 1.) * 10/epochs))] for j in range(10) ]) + '] [{}/{}] acc = {} % | loss = {} | AUC {} \t'.format(i+1, epochs, round(a / n, 4) * 100, round(l/n,3), AUC)
            CM_reporter = 'TP {}, TN {}, FP {} , FN {}, MCC : {}'.format(tp, tn, fp, fn, mcc)
            print(training_reporter + CM_reporter)
     
    model_specs['training_acc'] = a / n
