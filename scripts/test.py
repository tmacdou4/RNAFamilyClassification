from utility import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn import metrics
import pdb

def test (MODELNAME, dataloader, model_specs, device = 'cuda:0', foldn = 0):
    loss = model_specs['lossfn']
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
            loss_val = loss(out[:,0], y).view(-1, 1).expand_as(out)
            acc = torch.eq(out.argmax(dim = -1), y.long()).float().view(-1,1)
            a += float(acc.mean().detach().cpu().numpy())
            l += float(loss_val.mean())
            tn, fp, fn, tp = metrics.confusion_matrix(y.detach().cpu().numpy(), out.argmax(dim=-1).detach().cpu().numpy(), labels = np.arange(2, dtype=int)).ravel() 
            optimizer.zero_grad()
            loss_val.mean().backward()
            optimizer.step()
            #  print ('gradient step [ {} / {} ]'.format(n, model_specs['gr_steps']))
            training_reporter = 'FOLD {} STEP[{}/{}]\tTRAIN ['.format(str(foldn).zfill(3), n ,model_specs['gr_steps'] ) + ''.join([['#','.'][int(j > int((i + 1.) * 10/epochs))] for j in range(10) ]) + '] [{}/{}] {} % '.format(i+1, epochs, round(a / n, 4) * 100)
            CM_reporter = 'TP {}, TN {}, FP {} , FN {}, MCC : {}'.format(tp, tn, fp, fn, mcc)
            print(training_reporter + CM_reporter)
     
    model_specs['training_acc'] = a / n
