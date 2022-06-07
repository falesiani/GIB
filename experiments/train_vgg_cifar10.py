# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:42:05 2021

@author: Xi Yu
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os


from scipy.spatial.distance import pdist, squareform
import numpy as np

from model import cfg
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from mygate import ShannonEntropyLoss, RenyiEntropyLoss
from mygate import MyGate, Mask, MyDiscreteGate, MyDiscreteMyGate, FunctionMask, MyDiscrete, MyGateAttention


# define the model with gate layer
use_gate_flag = True
use_discrete_gate_flag = False
use_discrete_mygate_flag = True
use_discrete_flag = True
use_attention_flag = False

class VGG_gate(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_gate, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        if use_gate_flag:
            if use_discrete_gate_flag:
                self.gate3 = MyDiscreteGate(512)
            elif use_discrete_mygate_flag:
                self.gate3 = MyDiscreteMyGate(512)      
            elif use_attention_flag:
                self.gate3 = MyGateAttention(512,3)
            else:
                self.gate3 = MyGate(512)                
        else:
            self.mask = Mask()         
        
        self._loss_entropy = nn.CrossEntropyLoss()
        
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
    def get_mask_count(self):
        if use_gate_flag:
            return (self.get_mask()>0.5).long().sum()
        else:
            return 0.
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
        out = self.features(x)
        z = out.view(out.size(0), -1)
        z = self.gate3()
        out = self.classifier(z)
        return z,out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



##--------------------CIFAR-10 dataset---------------------------------#
print('==> Preparing data..')
batch_size = 128
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=6)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def train(net,epoch,gamma):

    print('\nEpoch [{}/{}]'.format(epoch+1, args.epochs))
    net.train()
    train_loss = 0
    IXZ_loss = 0
    HZ_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        Z,outputs = net(inputs)
        loss = criterion(outputs, targets)
        with torch.no_grad():
            Z_numpy = Z.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :1], 1))) 
            sigma = max(1e-5,sigma) 
            
            
        HZ  = reyi_entropy(Z,s_x=sigma**2)

        
        total_loss = loss+gamma*HZ
        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        HZ_loss += HZ.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print ('Step [{}/{}], Loss: {:.4f}, I_xz: {:.4f}, H_z: {:.4f}, Acc: {}% [{}/{}])' 
               .format(batch_idx, 
                       len(train_loader), 
                       train_loss/(batch_idx+1),
                       HZ_loss/(batch_idx+1),
                       100.*correct/total, correct, total))
    return HZ_loss/(batch_idx+1),train_loss/(batch_idx+1),100.*correct/total
    

def test(net,epoch,nbatches):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
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

    print("mask(test)={}".format(net.get_mask_count()))
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
        torch.save(state, './checkpoint/ckpt_std_IG.pth')
        best_acc = acc
    return test_loss/(batch_idx+1),100.*correct/tota

#main function
def main():
    best_acc = 0
    gamma = 1e-1
    epochs = 100
    all_IB_acc = []
    
    print('==> Building model..')
    if True:
        net = VGG_gate("VGG16")
    else:
        net = VGG_gate("VGG1")
        print("ATTENTION, this is a smaller version of VGG")
        
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)


    for epoch in range(epochs):
        train(net,epoch,gamma)
        acc = test(epoch)
        scheduler.step()















