#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file:
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
#  *   Copyright (c) 2020 NEC Laboratories Europe GmbH. All Rights Reserved.
#  *
#  * Authors: Francesco Alesiani  francesco.alesiani@neclab.eu
#  *
#  * 2019 NEC Laboratories Europe GmbH. DISCLAIMS ALL 
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

class FunctionMask(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        return (mask >= 0).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

def attention(q, k, d_k):
#     F.softmax(Y0 @ Y1.transpose(),dim=1)
    q = F.normalize(q, dim=-1, p=2)
    k = F.normalize(k, dim=-1, p=2)
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    return scores    
    
class Mask(nn.Module):
#     @staticmethod
    def forward(ctx, mask):
        return FunctionMask.apply(mask)
#         return (mask >= 0).float()

#     @staticmethod
#     def backward(ctx, g):
#         # send the gradient g straight-through on the backward pass.
#         return g    
    
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
    
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)

def renyi_entropy(x,sigma):
    alpha = 1.01
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
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
        H = renyi_entropy(x,sigma=sigma**2)        
        return H    
    

class MyDiscreteGate(Module):
    r"""Applies a linear gating transformation to the incoming data: :math:`y = x * sigmoid(w) `
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Parameter

    def __init__(self, out_features: int, use_uniform_flag:bool = True) -> None:
        super(MyDiscreteGate, self).__init__()
        self.in_features = out_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features))        
        self.use_uniform_flag=use_uniform_flag
        self.reset_parameters(use_uniform_flag)

        def hook_fn(module, input, output):            
            if False:
                print(f"input={(input[-1]>0.5).long().sum(axis=1)}")
            if type(output) in [tuple,list]:
                print(f"output{output[0].shape[0]}={(output[0]!=0).long().sum(axis=1)}")
            else:
                print(f"output{output.shape[0]}={(output>0.5).long().sum(axis=1)}")

        self.hook_fn = hook_fn
        if False:
            self.register_backward_hook(self.hook_fn)
            self.register_forward_hook(self.hook_fn)

    def reset_parameters(self,use_uniform_flag) -> None:
#         init.uniform_(self.weight, -10, 10)
#         init.uniform_(self.weight, .1, 2)   
#         init.constant_(self.weight, .1)

        if use_uniform_flag:
#             init.uniform_(self.weight, -1e-2, 1e-2)   
            init.uniform_(self.weight, -1e-1, 1e-1)   
        else:
            init.constant_(self.weight, 1e-1)   
    
    def update_mask(self):
        return FunctionMask.apply(self.weight)
    
    def get_mask(self):
        return FunctionMask.apply(self.weight)
    
    def get_gates(self):
        return FunctionMask.apply(self.weight)   
        
    def forward(self, input: Tensor) -> Tensor:
        return FunctionMask.apply(self.weight) * input

    def get_loss(self):
        return 1e0*torch.sqrt(((torch.abs(FunctionMask.apply(self.weight)) - torch.ones_like(self.weight))**2).sum())

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


    
class MyDiscreteMyGate(Module):
    r"""Applies a linear gating transformation to the incoming data: :math:`y = x * sigmoid(w) `
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, out_features: int) -> None:
        super(MyDiscreteMyGate, self).__init__()
        self.in_features = out_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features))        
        self.mask = Parameter(torch.ones_like(self.weight))
        self.mask.requires_grad = False
        self.reset_parameters()

        def hook_fn(module, input, output):            
            if False:
                print(f"input={(input[-1]>0.5).long().sum(axis=1)}")
            if type(output) in [tuple,list]:
                print(f"output{output[0].shape[0]}={(output[0]!=0).long().sum(axis=1)}")
            else:
                print(f"output{output.shape[0]}={(output>0.5).long().sum(axis=1)}")

        self.hook_fn = hook_fn
        if False:
            self.register_backward_hook(self.hook_fn)
            self.register_forward_hook(self.hook_fn)

    def reset_parameters(self) -> None:
#         init.uniform_(self.weight, -10, 10)
#         init.uniform_(self.weight, .1, 2)   
#         init.constant_(self.weight, 10.)
        init.uniform_(self.weight, -.1, .1)

    def update_mask(self):
        def get_mask_(a,m,th=2,th_a=.33):
            b = 1 - (-th_a < a ).float()*(a < th_a).float() 
            if sum(b*m)<th:
                return m
            else: 
                return b*m
#         values = self.gate_fn(self.weight).detach()
        values = FunctionMask.apply(self.weight).detach()        
        self.mask *=  get_mask_(values,self.mask)
        return self.mask
    
    def get_mask(self):
        return self.mask
    
    def get_gates(self):
        return FunctionMask.apply(self.weight)    
        
    def forward(self, input: Tensor) -> Tensor:
        return self.mask * FunctionMask.apply(self.weight) * input

    def get_loss(self):
        return 1e0*torch.sqrt(((torch.abs(FunctionMask.apply(self.weight)) - torch.ones_like(self.weight))**2).sum())

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class MyDiscrete(Module):
    r"""Applies a linear gating transformation to the incoming data: :math:`y = x * sigmoid(w) `
    """
    def forward(self, mask_: Tensor, input: Tensor) -> Tensor:
        return FunctionMask.apply(mask_) * input
    
    
class MyGate(Module):
    r"""Applies a linear gating transformation to the incoming data: :math:`y = x * sigmoid(w) `
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, out_features: int, use_tanh: bool = True) -> None:
        super(MyGate, self).__init__()
        self.in_features = out_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features))        
        self.mask = Parameter(torch.ones_like(self.weight))
        self.mask.requires_grad = False
        if use_tanh:
            self.gate_fn = torch.tanh
        else:
            self.gate_fn = torch.sigmoid
        self.reset_parameters()

        def hook_fn(module, input, output):            
            if False:
                print(f"input={(input[-1]>0.5).long().sum(axis=1)}")
            if type(output) in [tuple,list]:
                print(f"output{output[0].shape[0]}={(output[0]!=0).long().sum(axis=1)}")
            else:
                print(f"output{output.shape[0]}={(output>0.5).long().sum(axis=1)}")

        self.hook_fn = hook_fn
        if False:
            self.register_backward_hook(self.hook_fn)
            self.register_forward_hook(self.hook_fn)

    def reset_parameters(self) -> None:
#         init.uniform_(self.weight, -10, 10)
#         init.uniform_(self.weight, .1, 2)   
        init.constant_(self.weight, 10.)
    

    def update_mask(self):
        def get_mask_(a,m,th=2,th_a=.33):
            b = 1 - (-th_a < a ).float()*(a < th_a).float() 
            if sum(b*m)<th:
                return m
            else: 
                return b*m
        values = self.gate_fn(self.weight).detach()
        self.mask *=  get_mask_(values,self.mask)
        return self.mask
    
    def get_mask(self):
        return self.mask
    
    def get_gates(self):
        return self.gate_fn(self.weight)    
        
    def forward(self, input: Tensor) -> Tensor:
        return self.mask * self.gate_fn(self.weight) * input

    def get_loss(self):
        return 1e0*torch.sqrt(((torch.abs(self.gate_fn(self.weight)) - torch.ones_like(self.weight))**2).sum())

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class MyGateAttention(Module):
    r"""Applies a linear gating transformation to the incoming data: :math:`y = x * sigmoid(w) `
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    hidden_dim: int
    weight: Tensor

    def __init__(self, out_features: int, hidden_dim: int, use_tanh: bool = True) -> None:
        super(MyGateAttention, self).__init__()
        self.in_features = out_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        
#         self.ks = Parameter(torch.Tensor(out_features*hidden_dim))
#         self.qs = Parameter(torch.Tensor(out_features*hidden_dim))
        self.ks = nn.Linear(out_features, out_features*hidden_dim)
        self.qs = nn.Linear(out_features, out_features*hidden_dim)
        self.tau = 1.
        
#         self.reset_parameters()

        def hook_fn(module, input, output):            
            if False:
                print(f"input={(input[-1]>0.5).long().sum(axis=1)}")
            if type(output) in [tuple,list]:
                print(f"output{output[0].shape[0]}={(output[0]!=0).long().sum(axis=1)}")
            else:
                print(f"output{output.shape[0]}={(output>0.5).long().sum(axis=1)}")

        self.hook_fn = hook_fn
        if False:
            self.register_backward_hook(self.hook_fn)
            self.register_forward_hook(self.hook_fn)

    def reset_parameters(self) -> None:
#         init.uniform_(self.weight, -10, 10)
#         init.uniform_(self.weight, .1, 2)   
#         init.constant_(self.ks, 10.)
        init.uniform_(self.ks, 1., 10.)
        init.uniform_(self.qs, 1., 10.)
    

    def update_mask(self):
        return None
#         def get_mask_(a,m,th=2,th_a=.33):
#             b = 1 - (-th_a < a ).float()*(a < th_a).float() 
#             if sum(b*m)<th:
#                 return m
#             else: 
#                 return b*m
#         values = self.gate_fn(self.weight).detach()
#         self.mask *=  get_mask_(values,self.mask)
#         return self.mask
    
    def get_mask(self):
        return None        
#         return self.mask
    
    def get_gates(self):
#         return self.gate_fn(self.weight)    
        return None        
        
    def forward(self, input: Tensor) -> Tensor:
#         return self.mask * self.gate_fn(self.weight) * input
        q = self.qs(input)
        k = self.ks(input)
#         print(f"q.shape={q.shape}")
        _shape = list(q.shape)
        _shape = [_shape[0]]+[self.out_features]+[self.hidden_dim]
        q = q.reshape(_shape)
        k = k.reshape(_shape)
        scores = attention(q, k, self.tau)
        print(f"input.shape={input.shape},scores.shape = {scores.shape}")
        return input * scores

    def get_loss(self):
#         return 1e0*torch.sqrt(((torch.abs(self.gate_fn(self.weight)) - torch.ones_like(self.weight))**2).sum())
        return 0.

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


