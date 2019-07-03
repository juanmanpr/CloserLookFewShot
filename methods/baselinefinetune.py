import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax", 
                       init_orthogonal = False, ortho_reg = None, device = None, wd = 1e-3):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type
        self.ortho = ortho_reg
        self.device = device
        self.init_ortho = init_orthogonal        
        self.wd = wd

    def set_forward(self,x,is_feature = True):
        return self.set_forward_adaptation(x,is_feature); #Baseline always do adaptation
 
    def set_forward_adaptation(self,x,is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = y_support.to(self.device)

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way, self.init_ortho)
        linear_clf = linear_clf.to(self.device)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, self.wd)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(self.device)
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                
                if self.ortho:
                    if self.loss_type == 'dist': #Baseline ++
                        oloss =  l2_reg_ortho(linear_clf.L.weight, self.device)
                    else:
                        oloss =  l2_reg_ortho(linear_clf.weight, self.device)
                else:
                    oloss = 0.0                  
                    
                loss = loss + 1e-4*oloss
                
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
        
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
