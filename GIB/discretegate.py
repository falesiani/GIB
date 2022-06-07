#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: discretegate.py
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
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import torch.nn as nn
from torch import autograd
import math

from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch


class FunctionMask(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        return (mask >= 0).float()
    @staticmethod
    def backward(ctx, g):
        return g
    
class Mask(nn.Module):
    def forward(ctx, mask):
        return FunctionMask.apply(mask)
    
def get_masked(x,mask):
    hardmask = FunctionMask.apply(mask)
    return x * hardmask

class ShannonEntropyLoss(nn.Module):
    def __init__(self):
        super(ShannonEntropyLoss, self).__init__()
    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(1e-6+x, dim=1)
        b = -1.0 * b.sum()
        return b
    
def pairwise_distances(x):
    assert(len(x.shape)==2), "x should be two dimensional"
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)

def renyi_entropy(x,sigma):
    alpha = 1.01
    sigma = max(1e-4,sigma)
    k = calculate_gram_mat(x,sigma)    
    k = k/(1e-6+torch.trace(k)) 
    eigv = torch.abs(torch.linalg.eigvalsh(k))
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy

class RenyiEntropyLoss(nn.Module):
    def __init__(self):
        super(RenyiEntropyLoss, self).__init__()
    def forward(self, x):
        with torch.no_grad():
            x_numpy = x.cpu().detach().numpy()
            k = squareform(pdist(x_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
            sigma = max(1e-4,sigma)
        H = renyi_entropy(x,sigma=sigma**2)        
        return H    
    
class DiscreteGate(Module):
    r"""
    Applies soft/hard gating function: :math:`y = m * fn(w) `
    with 
        fn = sigmoid, tanh, or step
        m is the mask
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, out_features: int, mask_type: str = 'tanh', th: float = 10, th_a: float =.33, init_: str='uniform', const_: float=.1) -> None:
        super(DiscreteGate, self).__init__()
        self.mask_type = mask_type
        self.th = th
        self.th_a= th_a
        self.in_features = out_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features))        
        self.mask = Parameter(torch.ones_like(self.weight))
        self.mask.requires_grad = False
        assert(init_ in ['uniform','uniform-plus','cont'])
        self.init_ = init_
        self.const_ = const_
        assert(mask_type in ['tanh','sigmoid','discrete'])
        if mask_type in ['tanh']:
            self.gate_fn = torch.tanh
        if mask_type in ['sigmoid']:
            self.gate_fn = torch.sigmoid        
        if mask_type in ['discrete']:
            self.gate_fn = FunctionMask.apply
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_  in ['uniform']:
            init.uniform_(self.weight, -self.const_, self.const_)
        if self.init_  in ['uniform-plus']:
            init.uniform_(self.weight, 0, self.const_)
        if self.init_  in ['const']:
            init.constant_(self.weight, self.const_)
    
    def update_mask(self):
        def get_mask_(a,m,th,th_a):
            b = 1 - (-th_a < a ).float()*(a < th_a).float() 
            if sum(b*m)<th:
                return m
            else: 
                return b*m
        values = self.gate_fn(self.weight).detach()
        self.mask *=  get_mask_(values,self.mask,self.th,self.th_a)
        return self.mask
    
    def get_mask(self):
        return self.mask
    
    def get_gates(self):
        return self.gate_fn(self.weight)    
        
    def forward(self, input: Tensor) -> Tensor:
        return self.mask * self.gate_fn(self.weight) * input
        # return self.gate_fn(self.weight) * input
    
    def get_loss(self):
        return 1e0*torch.sqrt(((torch.abs(self.gate_fn(self.weight)) - torch.ones_like(self.weight))**2).sum())

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)
