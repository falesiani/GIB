# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:58:57 2021

@author: Xi Yu
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torchvision import datasets, transforms
from mygate import ShannonEntropyLoss, RenyiEntropyLoss


import numpy as np
import os

#FA
from mygate import EntropyLoss, MyGate


# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#mnist dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../.', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),

    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../.', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=128, shuffle=True)


class ConvNetGate(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetGate, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(7*7*32, num_classes)
        
        self.gate3 = MyGate(7*7*32)
        self._loss_entropy = EntropyLoss()
        
        self.pre_encoder = nn.Sequential(self.layer1, self.layer2, nn.Flatten())
        self.encoder = nn.Sequential(self.pre_encoder, self.gate3)
        
    def get_features_net(self):
        return self.pre_encoder
    def get_gate_net(self):
        return self.gate3      
    def get_classifier_net(self):
        return self.decoder_logits
    def update_mask(self):
        self.gate3.update_mask()
    def get_mask(self):
        return self.gate3.get_mask()
    def get_gates(self):
        return self.gate3.get_gates()
    def get_energy(self):
        return sum([p.detach().abs().mean() for p in self.parameters()])
    def get_entropy_loss(self,x):
        return self._loss_entropy(self.pre_encoder(x))
    def get_gated_entropy_loss(self,x):
        return self._loss_entropy(self.encoder(x))
    def get_gate_loss(self, x, phase_flag):
        if phase_flag:
            loss_ = self.get_entropy_loss(x)
        else:
            loss_ = self.get_gated_entropy_loss(x)    
        return loss_
        
    def forward(self, x):
        encoder = self.encoder(x)
        output = self.fc(encoder)
        return encoder, output
    
    
    class MLPGate(nn.Module):
        def __init__(self, num_classes):
            super(MLPGate, self).__init__()
                    
            self.fc1 = nn.Linear(784, 1024)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 256)
            self.fc4 = nn.Linear(256,num_classes)
            self.gate3 = MyGate(256)
            self._loss_entropy = EntropyLoss()
            self.pre_encoder = nn.Sequential(self.fc1
                                             ,self.relu
                                             ,self.fc2
                                             ,self.relu
                                             ,self.fc3)
            self.encoder = nn.Sequential(self.pre_encoder, self.gate3)
            
        def get_features_net(self):
            return self.pre_encoder
        def get_gate_net(self):
            return self.gate3      
        def get_classifier_net(self):
            return self.decoder_logits
        def update_mask(self):
            self.gate3.update_mask()
        def get_mask(self):
            return self.gate3.get_mask()
        def get_gates(self):
            return self.gate3.get_gates()
        def get_energy(self):
            return sum([p.detach().abs().mean() for p in self.parameters()])
        def get_entropy_loss(self,x):
            return self._loss_entropy(self.pre_encoder(x))
        def get_gated_entropy_loss(self,x):
            return self._loss_entropy(self.encoder(x))
        def get_gate_loss(self, x, phase_flag):
            if phase_flag:
                loss_ = self.get_entropy_loss(x)
            else:
                loss_ = self.get_gated_entropy_loss(x)    
            return loss_
        
        def forward(self, x):
            encoder = self.encoder(x)
            output = self.fc4(encoder)
            return encoder, output
    
    
    # Training
def train(net,epoch,criterion,optimizer,beta):
    #print('\nEpoch: %d' % epoch)
    print('\nEpoch [{}/{}]'.format(epoch+1, 200))
    net.train()
    train_loss = 0
    XZ_loss = 0
    ZY_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #inputs = inputs.to(device)
        inputs = inputs.view(-1,784)
        target = F.one_hot(targets,10)
        inputs = inputs.to(device)
        target = target.to(device).float()
        targets = targets.to(device)
        Z,outputs = net(inputs)
        with torch.no_grad():
            Z_numpy = Z.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1))) 
            
        IXZ = RenyiEntropyLoss(inputs,Z,s_x=1000,s_y=sigma**2)
        IZY = RenyiEntropyLoss(Z,target,s_x=sigma**2,s_y=2)
        cross_entropy_loss = criterion(outputs, targets)
        loss = cross_entropy_loss+beta*IXZ

        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        XZ_loss += IXZ.item()
        ZY_loss += IZY.item()
        train_loss += cross_entropy_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print ('Step [{}/{}], ce_Loss: {:.4f},xz_Loss: {:.4f}, Acc: {}% [{}/{}]),sigma:{:.4f}' 
               .format(batch_idx, 
                       len(train_loader), 
                       train_loss/(batch_idx+1),
                       XZ_loss/(batch_idx+1),
                       100.*correct/total, correct, total,sigma))
                
    return XZ_loss/(batch_idx+1),ZY_loss/(batch_idx+1),train_loss/(batch_idx+1),100.*correct/total
       


def test(net,epoch,criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #inputs = inputs.to(device)
            inputs = inputs.reshape(-1, 28 * 28).to(device)
            targets =  targets.to(device)
            _,outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print ('Step [{}/{}], Loss: {:.4f}, Acc: {}% [{}/{}])' 
                   .format(batch_idx, 
                           len(test_loader), 
                           test_loss/(batch_idx+1),
                           100.*correct/total, correct, total))
        

    

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_mnist'+'.pth')
        best_acc = acc
    return test_loss/(batch_idx+1),100.*correct/total


#main function
import time
net = MLPGate(num_classes=10)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
beta = 1e-6
print(beta)
train_MI_XZ = []
train_MI_YZ = []
train_loss = []
train_Accuracy = []
test_mse_Loss = []
test_Acc = []
best_acc = 0
for epoch in range(1):
    if epoch % 2 == 0 and epoch > 0:
        scheduler.step()
    start_time = time.time()
    train_xz,train_yz,loss_train,train_accuracy = train(beta,net,epoch)
    test_mse_loss,test_accuracy = test(net,epoch)

    print(time.time() - start_time)

    train_MI_XZ.append(train_xz)
    train_MI_YZ.append(train_yz)
    train_loss.append(loss_train)
    train_Accuracy.append(train_accuracy)
    test_mse_Loss.append(test_mse_loss)
    test_Acc.append(test_accuracy)















