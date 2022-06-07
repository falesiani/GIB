##################
#Adapted from https://github.com/facebookresearch/GradientEpisodicMemory. 


# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree (https://github.com/facebookresearch/GradientEpisodicMemory).


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
import quadprog

class GEMMLP(nn.Module):
    def type_(self): return "gem"
    def __init__(self, length, width, height, hidden_dim, num_classes, dropp, n_memories , n_tasks ,regression=False):
        super(GEMMLP, self).__init__()
        self.regression=regression                
        
        ft1 = nn.Flatten()
        lin1 = nn.Linear(width*length*height, hidden_dim)
        do1 = nn.Dropout(dropp)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        do2 = nn.Dropout(dropp)
        lin3 = nn.Linear(hidden_dim, num_classes)
#         self._loss = nn.BCEWithLogitsLoss()
        self._loss = nn.CrossEntropyLoss()
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()
    
        self._main = nn.Sequential(ft1, lin1,do1, nn.ELU(True), lin2,do2, nn.ELU(True), lin3)
        
        # setup memories
        self.n_memories = n_memories
        self.n_inputs  = length*width*height
        if self.regression:
            self.in_shape = [length, width,  height]
        else:
            self.in_shape = [width, length, height]            
        self.n_outputs = num_classes        
        self.n_tasks = n_tasks
               
        # allocate episodic memory
        self.memory_data = torch.FloatTensor(self.n_tasks, self.n_memories, *self.in_shape)
        if self.regression:
            self.memory_labs = torch.LongTensor(self.n_tasks, self.n_memories, self.n_outputs)
        else:
            self.memory_labs = torch.LongTensor(self.n_tasks, self.n_memories, 1)
            
        
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_tasks)

        self.gpu = False
        if torch.cuda.is_available():
            self.gpu = True
            self.cuda()
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()        
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.nc_per_task = self.n_outputs        
        

    @staticmethod
    def store_grad(pp, grads, grad_dims, tid):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
            tid: task id
        """
        # store the gradients
        grads[:, tid].fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en, tid].copy_(param.grad.data.view(-1))
            cnt += 1

    @staticmethod
    def overwrite_grad(pp, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1

    @staticmethod
    def project2cone2(gradient, memories, margin=0.5):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))
        
        
    def forward(self, x):
        out = self._main(x)
        return out
    
    def get_cl_loss(self,t,x,y):
        self.train()
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            
        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt

        self.memory_data[t, self.mem_cnt: endcnt].copy_(x.data[:effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(y.data[:effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                _y = self.forward(Variable(self.memory_data[past_task]))
                ptloss = self.eval_loss(_y,Variable(self.memory_labs[past_task]))
                ptloss.backward()
                self.store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()
        
        _y = self.forward(x)
        loss = self.eval_loss(_y, y)

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            self.store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                self.project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                self.overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)                        
            
        return loss
    
    def get_energy(self):
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def eval_loss(self,y_,y):
#         labels = y.view([-1]).long()         
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels)
#         return self._loss(y_, onehot(y))



import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module


#From an implementation of MER Algorithm 1 from https://openreview.net/pdf?id=B1gTShAct7
# or https://github.com/mattriemer/mer

# An implementation of MER Algorithm 1 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree of https://github.com/mattriemer/mer.

class MERMLP(nn.Module):
    def type_(self): return "mer"
    def __init__(self, length, width, height, hidden_dim, num_classes, dropp, n_memories , n_tasks, replay_batch_size, batches_per_example, beta, gamma ,regression=False):
        super(MERMLP, self).__init__()
        self.regression=regression                
        ft1 = nn.Flatten()
        lin1 = nn.Linear(width*length*height, hidden_dim)
        do1 = nn.Dropout(dropp)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        do2 = nn.Dropout(dropp)
        lin3 = nn.Linear(hidden_dim, num_classes)
#         self._loss = nn.BCEWithLogitsLoss()
        self._loss = nn.CrossEntropyLoss()
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()
    
        self.bce = CrossEntropyLoss()
        self._main = nn.Sequential(ft1, lin1,do1, nn.ELU(True), lin2,do2, nn.ELU(True), lin3)
  
        self.n_inputs  = length*width*height
        if self.regression:
            self.in_shape = [length, width, height]
        else:
            self.in_shape = [width, length, height]
        self.n_outputs = num_classes        
        self.n_tasks = n_tasks


        self.batchSize = int(replay_batch_size)

        self.memories = n_memories
        self.steps = int(batches_per_example)
        self.beta = beta
        self.gamma = gamma

        # allocate buffer
        self.M = []
        self.age = 0

        self.gpu = False
        if torch.cuda.is_available():
            self.gpu = True
            self.cuda()

    def forward(self, x, t):
        output = self.net(x)
        return output

    def getBatch(self,x,y,t):
        xi = Variable(torch.from_numpy(np.array(x))).float().reshape([1,-1])
        if self.regression:
            yi = Variable(torch.from_numpy(np.array(y))).float().reshape([1,-1])
        else:
            yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
        if self.gpu:
            xi = xi.cuda()
            yi = yi.cuda()
            
        bxs = [xi]
        bys = [yi]
        
        if len(self.M) > 0:
            order = [i for i in range(0,len(self.M))]
            osize = min(self.batchSize,len(self.M))
            for j in range(0,osize):
                shuffle(order)
                k = order[j]
                x,y,t = self.M[k]
                xi = Variable(torch.from_numpy(np.array(x[0]))).float().reshape([1,-1])
                if self.regression:
                    yi = Variable(torch.from_numpy(np.array(y[0]))).float().reshape([1,-1])
                else:    
                    yi = Variable(torch.from_numpy(np.array(y[0]))).long().view(1)
                # handle gpus if specified
                if self.gpu:
                    xi = xi.cuda()
                    yi = yi.cuda()
                bxs.append(xi)
                bys.append(yi)                
 
        return bxs,bys
        
    def forward(self, x):
        out = self._main(x)
        return out
    
    def step(self, t, x, y, opt, batch_size):
        self.train()
        ### step through elements of x
        for i in range(0,x.size()[0],batch_size):
            self.age += 1
            _from,_to = i*batch_size,(i+1)*batch_size
            xi = x[_from:_to].data.cpu().numpy()
            yi = y[_from:_to].data.cpu().numpy()
            if len(xi)==0: break
            self._main.zero_grad()

            before = deepcopy(self._main.state_dict())
            for step in range(0,self.steps):                
                weights_before = deepcopy(self._main.state_dict())
                # Draw batch from buffer:
                bxs,bys = self.getBatch(xi[0],yi[0],t)
                loss = 0.0
                for idx in range(len(bxs)):
                    self._main.zero_grad()
                    _xi = torch.Tensor(xi[1:]).cuda()
                    _yi = torch.Tensor(yi[1:]).cuda()
                    bx = torch.cat([bxs[idx].reshape([1,*self.in_shape]),_xi],0)
                    if self.regression:
                        by = torch.cat([bys[idx].reshape([1,-1]).float(),_yi.float()],0)                    
                    else:
                        by = torch.cat([bys[idx].reshape([1,1]).long(),_yi.long()],0)                    
                    prediction = self.forward(bx)
                    loss = self.eval_loss(prediction, by)
                    loss.backward()
                    opt.step()
                
                weights_after = self._main.state_dict()
                
                # Within batch Reptile meta-update:
                self._main.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before})
            
            after = self._main.state_dict()
            
            # Across batch Reptile meta-update:
            self._main.load_state_dict({name : before[name] + ((after[name] - before[name]) * self.gamma) for name in before})
                    

            # Reservoir sampling memory update: 
            
            if len(self.M) < self.memories:
                self.M.append([xi,yi,t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi,yi,t]    
    
    def get_energy(self):
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def eval_loss(self,y_,y):
#         labels = y.view([-1]).long()         
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels)
#         return self._loss(y_, onehot(y))

#################
#adapted from ewc.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree (https://github.com/facebookresearch/GradientEpisodicMemory).

class EWCMLP(nn.Module):
    def type_(self): return "ewc"
    def __init__(self,length,width,height, hidden_dim,num_classes,dropp=.75,n_memories=1,reg=1.,regression=False):
        super(EWCMLP, self).__init__()
        self.regression=regression                
        
        ft1 = nn.Flatten()
        lin1 = nn.Linear(length*width*height, hidden_dim)
        do1 = nn.Dropout(dropp)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        do2 = nn.Dropout(dropp)
        lin3 = nn.Linear(hidden_dim, num_classes)
#         self._loss = nn.BCEWithLogitsLoss()
        self._loss = nn.CrossEntropyLoss()
        self._loss = nn.L1Loss() if regression else nn.CrossEntropyLoss()    
        self._main = nn.Sequential(ft1, lin1,do1, nn.ELU(True), lin2,do2, nn.ELU(True), lin3)
        
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None
        self.n_memories = n_memories
        self.reg = reg
        
    def forward(self, x):
        out = self._main(x)
        return out
    
    def get_cl_loss(self,t,x,y):
        self.train()
        if t != self.current_task:
            self.fisher[self.current_task] = []
            self.optpar[self.current_task] = []
            for p in self.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher[self.current_task].append(pg)
            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]  
                    
        y_=self(x)
        loss = self.eval_loss(y_, y)
        for tt in range(t):
            for i, p in enumerate(self.parameters()):
                l = self.reg * Variable(self.fisher[tt][i])
                l = l * (p - Variable(self.optpar[tt][i])).pow(2)
                loss += l.sum()        
        return loss
    
    def get_energy(self):
#         return sum([p.abs().mean() for p in self.parameters()])   
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def eval_loss(self,y_,y):
#         labels = y.view([-1]).long()        
        labels = y if self.regression else y.view([-1]).long()
        return self._loss(y_, labels)
#         return self._loss(y_, onehot(y))

