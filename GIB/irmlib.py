#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: irmlib.py
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


import scipy as sp
import numpy as np
from numpy import matmul, trace, diag, real, ones, eye,log
import sys, os
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import loadmat
import joblib
import warnings
warnings.filterwarnings("ignore")
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import Module

import random
from random import shuffle
from copy import deepcopy

from GIB.discretegate import RenyiEntropyLoss,ShannonEntropyLoss, DiscreteGate

################
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(x.size(0), -1)

#################

class StdMLP(nn.Module):
    def type_(self): return "std"
    def __init__(self,length,width,height, hidden_dim,num_classes,dropp=.75,regression=False):
        super(StdMLP, self).__init__()
        self.regression=regression                
        ft1 = nn.Flatten()
        lin1 = nn.Linear(length*width*height, hidden_dim)
        do1 = nn.Dropout(dropp)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        do2 = nn.Dropout(dropp)
        lin3 = nn.Linear(hidden_dim, num_classes)
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()
        self._main = nn.Sequential(ft1, lin1,do1, nn.ELU(True), lin2,do2, nn.ELU(True), lin3)        
    def forward(self, x):
        out = self._main(x)
        return out
    def get_energy(self):
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def eval_loss(self,y_,y):
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels)



#################

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

#################
def dist_models(model_list):
    def _dist_models(m1,m2):
        return sum([((p1-p2)**2).sum() for p1,p2 in zip(m1.parameters(),m2.parameters())])
    m1 = model_list[0]
    return [_dist_models(m1,m2) for m2 in model_list[1:]]


#################

class StdMLP_feature(nn.Module):
    def type_(self): return "std"
    def __init__(self,length,width,height, hidden_dim,num_classes,dropp=.75,regression=False):
        super(StdMLP_feature, self).__init__()
        self.regression=regression                
        ft1 = nn.Flatten()
        lin1 = nn.Linear(length*width*height, hidden_dim)
        self._loss = nn.CrossEntropyLoss()
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()
        self._main = nn.Sequential(ft1, lin1, nn.ELU(True))
    def forward(self, x):
        out = self._main(x)
        return out
    def get_energy(self): 
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def eval_loss(self,y_,y):
#         labels = y.view([-1]).long()   
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels)


class StdMLP_class(nn.Module):
    def type_(self): return "std"
    def __init__(self,length,width,height, hidden_dim,num_classes,dropp=.75,regression=False):
        super(StdMLP_class, self).__init__()
        self.regression=regression                
        ft1 = nn.Flatten()
        lin1 = nn.Linear(length*width*height, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        do2 = nn.Dropout(dropp)
        lin3 = nn.Linear(hidden_dim, num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()
        self._main = nn.Sequential(ft1, lin2,do2, nn.ELU(True), lin3)
    def forward(self, x):
        out = self._main(x)
        return out
    def get_energy(self):
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def eval_loss(self,y_,y):
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels)


#################
class GatedMLP_class(nn.Module):
    def type_(self): return "gated"
    def __init__(self,length,width,height, hidden_dim,num_classes,dropp=.75, gate_lambda=1e-2, mask_type='tanh',regression=False):
        super(GatedMLP_class, self).__init__()
        self.regression=regression        
        self.gate_lambda=gate_lambda
        self.ft1 = nn.Flatten()
        self.lin1 = nn.Linear(length*width*height, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.do2 = nn.Dropout(dropp)
        self.gate3 = DiscreteGate(hidden_dim, mask_type)
        self.lin3 = nn.Linear(hidden_dim, num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()
        self._features = nn.Sequential(self.ft1, self.lin2, self.do2, nn.ELU(True))
        self._classifier = nn.Sequential(self.gate3, self.lin3)
        self._main = nn.Sequential(self._features, self._classifier)        
    def update_mask(self):
        self.gate3.update_mask()
    def get_features_net(self):
        return self._features
    def get_gate_net(self):
        return self.gate3      
    def get_classifier_net(self):
        return self._classifier      
    def forward(self, x):
        out = self._main(x)
        return out
    def get_energy(self):
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def eval_loss(self,y_,y):
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels) + self.gate_lambda*self.gate3.get_loss()

class GatedMLP(nn.Module):
    def type_(self): return "gated"
    def __init__(self,length,width,height, hidden_dim,num_classes,dropp=.75, gate_lambda=0,mask_type='tanh',regression=False, entropy_type='shannon'):
        super(GatedMLP, self).__init__()
        self.regression=regression
        self.gate_lambda=gate_lambda
        self.ft1 = nn.Flatten()
        self.lin1 = nn.Linear(length*width*height, hidden_dim)
        self.do1 = nn.Dropout(dropp)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.do2 = nn.Dropout(dropp)
        self.gate3 = DiscreteGate(hidden_dim, mask_type)
        self._critic = nn.Linear(hidden_dim, hidden_dim)        
        self.lin3 = nn.Linear(hidden_dim, num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()
        assert(entropy_type.lower() in ['shannon','renyi'])
        self._loss_entropy = ShannonEntropyLoss() if entropy_type=='shannon' else RenyiEntropyLoss()    
        self._features = nn.Sequential(self.ft1, self.lin1, self.do2, nn.ELU(True), self.lin2, self.do2, nn.ELU(True))
        self._classifier = nn.Sequential(self.gate3, self.lin3)
        self._features_gated = nn.Sequential(self._features, self.gate3)
        self._critic_gated = nn.Sequential(self._features_gated, nn.ELU(True), self._critic, nn.ELU(True))
        self._main = nn.Sequential(self._features, self._classifier)
    def get_features_net(self):
        return self._features
    def get_critic_net(self):
        return self._critic_gated 
    def get_gate_net(self):
        return self.gate3      
    def get_classifier_net(self):
        return self._classifier
    def update_mask(self):
        self.gate3.update_mask()
    def get_mask(self):
        return self.gate3.get_mask()
    def get_gates(self):
        return self.gate3.get_gates()
    def get_classifier_energy(self):
        return sum([p.detach().abs().mean() for p in self.lin3.parameters()]) 
    def get_gate_energy(self):
        return sum([p.detach().abs().mean() for p in self.gate3.parameters()]) 
    def get_features_energy(self):
        return sum([p.detach().abs().mean() for p in self._features.parameters()])     
    def forward(self, x):
        out = self._main(x)
        return out
    def get_energy(self):
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def get_entropy_loss(self,x):
        return self._loss_entropy(self._features(x))
    def get_gated_entropy_loss(self,x):
        return self._loss_entropy(self._features_gated(x))
    def eval_loss(self,y_,y):
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels) + self.gate_lambda*self.gate3.get_loss()
    def eval_loss_debug(self,y_,y): 
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels)+self.gate_lambda*self.gate3.get_loss(),self._loss(y_, labels),self.gate_lambda*self.gate3.get_loss(), 0.
    
        


