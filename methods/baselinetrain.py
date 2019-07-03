import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax', 
                       init_orthogonal = False, ortho_reg = None, device = None):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, 
                                                    num_class, init_orthogonal)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.ortho = ortho_reg
        self.device = device
        
    def forward(self,x):
        x    = x.to(self.device)
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = y.to(self.device)
        
        if self.ortho:
            if loss_type == 'dist': #Baseline ++
                self.oloss =  l2_reg_ortho(self.classifier.L.weight, self.device)
            else:
                self.oloss =  l2_reg_ortho(self.classifier.weight, self.device)
        else:
            self.oloss = 0.0        
        
        return self.loss_fn(scores, y) + 1e-4 * self.oloss
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0

        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration


def l2_reg_ortho(weight_matrix, device):

    W = weight_matrix
    
    cols = W[0].numel()
    rows = W.shape[0]
    w1 = W.view(-1,cols)
    wt = torch.transpose(w1,0,1)
    if (rows > cols):
        m  = torch.matmul(wt,w1)
        ident = torch.eye(cols,cols,requires_grad=True).to(device)
    else:
        m = torch.matmul(w1,wt)
        ident = torch.eye(rows,rows,requires_grad=True).to(device)

    w_tmp = (m - ident)
    b_k = torch.rand(w_tmp.shape[1],1).to(device)

    v1 = torch.matmul(w_tmp, b_k)
    norm1 = torch.norm(v1,2)
    v2 = torch.div(v1,norm1)
    v3 = torch.matmul(w_tmp,v2)

    l2_reg = (torch.norm(v3,2))**2
                    
    return l2_reg
