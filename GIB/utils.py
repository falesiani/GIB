#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: utils.py
#  *
#  *    NEC Laboratories Europe GmbH. PROPRIETARY INFORMATION
#  *
#  * This software is supplied under the terms of a license agreement
#  * or nondisclosure agreement with NEC Laboratories Europe GmbH. and 
#  * may not becopied or disclosed except in accordance with the terms of that
#  * agreement. The software and its source code contain valuable 
#  * trade secrets and confidential information which have to be 
#  * maintained in confidence. 
#  * Any unauthorized publication, transfer to third parties or 
#  * duplication of the object or source code - either totally or in 
#  * part - is prohibited. 
#  *

#  *
#  *   Copyright (c) 2021 NEC Laboratories Europe GmbH. All Rights Reserved.
#  *
#  * Authors: Francesco Alesiani  francesco.alesiani@neclab.eu
#  *
#  * 2021 NEC Laboratories Europe GmbH. DISCLAIMS ALL 
#  * WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  * INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND THE ACCOMPANYING
#  * DOCUMENTATION.
#  *
#  * No Liability For Consequential Damages IN NO EVENT SHALL 2019 NEC 
#  * Laboratories Europe GmbH, NEC Corporation 
#  * OR ANY OF ITS SUBSIDIARIES BE LIABLE FOR ANY
#  * DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS
#  * OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF NEC Europe Ltd. HAS BEEN ADVISED OF THE
#  * POSSIBILITY OF SUCH DAMAGES.
#  *
#  *     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  */


import numpy as np
import argparse
import IPython.display as display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import pandas as pd
import sys
import copy
import copy as cp
from sklearn.model_selection import KFold

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from torch.autograd import Variable

################
#regression
def correlation_coefficient(preds, targets):
    """
    Correlation coefficient
    """
    mx = preds.mean()
    my = targets.mean()
    xm, ym = preds-mx, targets - my
    r_num = torch.mul(xm, ym).sum() 
    r_den = torch.sqrt( torch.mul(torch.square(xm).sum(), torch.square(ym).sum()))
    r = torch.divide(r_num, r_den + 1e-6)
    r = torch.clamp(r,max=1.0,min = -1.0)
    return r
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.logflag = False
        
    def forward(self, pred, actual):
        if self.logflag:
            pred, actual = torch.log(pred + 1), torch.log(actual + 1)
        return torch.sqrt(self.mse(pred, actual))

def metric_regression(logits,targets):
    rmse = RMSLELoss()
    rmse_ = rmse(logits, targets).item()
    pcor_ = correlation_coefficient(logits, targets).item()
    return rmse_,pcor_
#classification
def onehot(yt,nb_classes=None):
    if nb_classes is None:
        nb_classes=int(yt.max())+1
    y_onehot = torch.FloatTensor(int(yt.shape[0]), nb_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, yt.long(), 1)
    return y_onehot

def eval_accuracy(y_,yt):
    _, pb = torch.max(y_.data, 1, keepdim=False)
    rt = (pb.view([-1]) == yt.view([-1])).float().sum()/yt.size(0)
    return rt

def dist_models(model_list, last_is_feature=False):
    def _dist_models(m1,m2):
        return sum([((p1-p2)**2).sum() for p1,p2 in zip(m1.parameters(),m2.parameters())])
    m1 = model_list[0]
    if last_is_feature:
        return [_dist_models(m1,m2) for m2 in model_list[1:-1]]
    else:
        return [_dist_models(m1,m2) for m2 in model_list[1:]]

def combine_models(model_list, x, use_last_flag=False, last_is_feature=False):
    #just use the last
    if use_last_flag: 
        return model_list[-1](x)    
    #there is only one model
    if len(model_list)==1:
        return model_list[0](x)    
    #here we use the last model as feature extraction, 
    #then sum the output of the other models on this input
    if last_is_feature:
        z = model_list[-1](x)
        return sum([model_i(z) for model_i in model_list[:-1]])
    else:
        return sum([model_i(x) for model_i in model_list])

def set_models(model_list,train_flag=True):
    if not type(model_list)==list:
        if train_flag: 
            model_list.train()
        else:
            model_list.eval()
    else:
        for model_i in model_list:
            if train_flag: 
                model_i.train()
            else:
                model_i.eval()
def concat_envs(data_tuple_list):
    n_e  = len(data_tuple_list)                     # number of environments
    # combine the data from the different environments x_in: combined data from environments, y_in: combined labels from environments, e_in: combined environment indices from environments
    x_in = data_tuple_list[0][0]
    for i in range(1,n_e):
        x_c = data_tuple_list[i][0]
        x_in = np.concatenate((x_in, x_c), axis=0)
    y_in = data_tuple_list[0][1]
    for i in range(1,n_e):
        y_c = data_tuple_list[i][1]
        y_in = np.concatenate((y_in, y_c), axis=0)
    e_in = data_tuple_list[0][2]
    for i in range(1,n_e):
        e_c = data_tuple_list[i][2]
        e_in = np.concatenate((e_in, e_c), axis=0)
    return x_in,y_in,e_in
def shuffle_envs(data_tuple_list):
    n_e = len(data_tuple_list)
    datat_list = []
    for ke in range(n_e):
        x_e = data_tuple_list[ke][0]
        y_e = data_tuple_list[ke][1]
        datat_list.append(shuffle(x_e,y_e)) 
    return datat_list

################
#
################
#help functions

def mean_nll(logits, y):
    logits = F.log_softmax(logits, dim=1)
    return F.nll_loss(logits, y.squeeze().long())    
def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)
######
def cross_entropy_manual(y,y_pred):
    return nn.functional.binary_cross_entropy(y_pred, y)
def loss_n(model,x,e,y,w,k):
    index = torch.where(e==k)
    y1_ = model(x[index[0]])*w
    y1  = y[index[0]]
    return model.eval_loss(y1_,y1)
# gradient of cross entropy loss w.r.t w for environment e
def grad_norm_n(model,x,e,y,w,k):
    loss = loss_n(model,x,e,y,w,k)  
    grad = autograd.grad(loss, [w], create_graph=True)[0]
    return torch.sum(grad**2)

def loss_grad(loss,model):
    grad = autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    return grad

def loss_0(model, x, y):
    y_ = model(x)
    return model.eval_loss(y_,y)    

# sum of cross entropy loss and penalty 
def loss_total(model,x,e,y,w,gamma, n_e):
    loss0 = loss_0(model,x,y)            
    loss_penalty = 0.0
    for k in range(n_e):
        loss_penalty += gamma*grad_norm_n(model,x,e,y,w,k)
    return (loss0 + loss_penalty)     

def loss_total_debug(model,x,e,y,w,gamma, n_e):
    loss0 = loss_0(model,x,y)            
    loss_penalty = 0.0
    for k in range(n_e):
        loss_penalty += gamma*grad_norm_n(model,x,e,y,w,k)
    return (loss0 + loss_penalty), loss0, loss_penalty   

def torch_list_sum(a): return  torch.stack(a).sum(dim=0)


