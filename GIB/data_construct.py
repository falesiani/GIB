#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: data_construct.py
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


import copy as cp
from sklearn.model_selection import KFold

import joblib


import torch
import torchvision
import torchvision.transforms as transforms


#####################
#Help function to create dataset

def add_channel(images,islast=True):
    if islast:
        return np.concatenate([images,np.zeros([*images[:,:,:,0].shape,1])],axis=-1)
    else:
        lst = list(images[:,0,:,:].shape)
        _shape = [lst[0]]+[1]+lst[1:]
        _ch = np.zeros(_shape)
        return np.concatenate([images,_ch],axis=1)

def flip2(y,pc,pl):
    yp=np.mod(y+np.random.binomial(1,pl,(len(y),1)),2)
    z=np.mod(yp+np.random.binomial(1,pc,(len(y),1)),2)
    return yp,z

def rndshift2(y,pc,pl,n_classes):
    shift = np.random.randint(1,n_classes,(len(y),1))
    b = np.random.binomial(1,pl,(len(y),1))
    yp = np.mod(y+shift*b,n_classes)
    
    shift = np.random.randint(1,n_classes,(len(y),1))
    b = np.random.binomial(1,pc,(len(y),1))
    z = np.mod(yp+shift*b,n_classes)
    
    return yp,z

def get_red(x, flip=False,zero=True):
    fn = lambda x: 1-x if flip else x
    red = [fn(x),(0 if zero else 1)*np.ones(x.shape)]
    return np.concatenate(red,axis=-1)
def get_green(x, flip=False,zero=True):
    fn = lambda x: 1-x if flip else x
    green = [(0 if zero else 1)*np.ones(x.shape),fn(x)]
    return np.concatenate(green,axis=-1)

def get_red_th(x,th,flip=False,zero=True):
    fn = lambda x: 1-x if flip else x
    red = [fn(x)*(x>=th),fn(x)*(x<th)]
    return np.concatenate(red,axis=-1)
def get_green_th(x,th,flip=False,zero=True):
    fn = lambda x: 1-x if flip else x
    green = [fn(x)*(x<th),fn(x)*(x>=th)]
    return np.concatenate(green,axis=-1)


def get_img_rgb(r,g,b):
    m,n= r.shape[:2]
    return np.concatenate([r.reshape(m,n,1),g.reshape(m,n,1),b.reshape(m,n,1)],axis=-1)

def get_img_rg(r,g):
    m,n= r.shape[:2]
    return np.concatenate([r.reshape(m,n,1),g.reshape(m,n,1)],axis=-1)

import math
def image_split(n):
    maxiter = 1e6
    ns = max(0,int(math.sqrt(n))-1)
    nx,ny = ns,ns
    while (nx+1)*(ny+1)<n and maxiter>0:
        maxiter-=1
        if nx<=ny:
            nx+=1
        else:
            ny+=1
    return nx,ny

#####################
#Help function to create dataset

def create_environment_a(env,x,y,pc,pl,flip,zero):
    y = y.astype(int)
    num_samples=len(y)   
    yp,z = flip2(y,pc,pl)
    red = np.where(z==1)[0]
    green = np.where(z==0)[0]
    tsh = 0.5
    r = get_red_th(x[red,:],.5,flip,zero)
    g = get_green_th(x[green,:],.5,flip,zero)
    dataset =np.concatenate((r,g),axis=0)
    labels = np.concatenate((yp[red,:],yp[green,:]),axis=0)
    x_in,y_in,e_in = dataset,labels,np.ones((num_samples,1))*env
    return x_in,y_in,e_in    

def create_environment_b(env,x,y,pc,pl,flip,zero):
    y = y.astype(int)
    num_samples=len(y)   
    yp,z = flip2(y,pc,pl)
    red = np.where(z==1)[0]
    green = np.where(z==0)[0]
    r = get_red(x[red,:],flip,zero)
    g = get_green(x[green,:],flip,zero)
    dataset =np.concatenate((r,g),axis=0)
    labels = np.concatenate((yp[red,:],yp[green,:]),axis=0)
    x_in,y_in,e_in = dataset,labels,np.ones((num_samples,1))*env
    return x_in,y_in,e_in    

def get_mask_color(x,k,n):
    assert(len(x.shape)==4), "x shall be [n_samples, hight, width, 1]"
    assert(x.shape[3]==1), "x shall be [n_samples, hight, width, 1]"
#     invert = 1 if invert else 0
#     x1 = np.ones(x.shape)*invert
    nx,ny = image_split(n)
    n1,n2 = x.shape[1:3]
    nxpix = int(n1/(nx+1)) if nx>0 else n1
    nxpiy = int(n2/(ny+1)) if ny>0 else n2
    ki = k % (nx+1)
    kj = int((k-ki)/(nx+1))% (ny+1)
#     print(ki,kj)
    f1,t1 = nxpix*ki,nxpix*(ki+1)
    f2,t2 = nxpiy*kj,nxpiy*(kj+1)
    x[:,f1:t1,f2:t2,0] = 1
    return x

def create_environment_d(env,x,y,pc,pl,flip,env_only,n_classes=10):
    y = y.astype(int)
    num_samples=len(y)   
    yp,z = rndshift2(y,pc,pl,n_classes) 

    imgs=[]
    labels=[]
    for kc in range(n_classes):
        idx = np.where(z==kc)[0]
        if kc%2==0:
            r = x[idx,:]
            pos = 2*env if env_only else kc+env
            if flip:
                g = np.zeros(r.shape)
                g = get_mask_color(g,pos,n_classes)
            else:
                r = get_mask_color(r,pos,n_classes)
                g = np.zeros(r.shape)
        else:
            g = x[idx,:]
            pos = 2*env+1 if env_only else kc+env
            if flip:
                r = np.zeros(g.shape)
                r = get_mask_color(r,pos,n_classes)
            else:
                g = get_mask_color(g,pos,n_classes)
                r = np.zeros(g.shape)
        labels += [np.mod(yp[idx,:],2)]
        imgs += [np.concatenate((r,g),axis=-1)]
    
    x_in = np.concatenate(imgs,axis=0)
    y_in = np.concatenate(labels,axis=0)
    e_in = np.ones((num_samples,1))*env
    return x_in,y_in,e_in    


def create_environment_original(env,x,y,pc,pl):
    y = y.astype(int)
    num_samples=len(y)   
    yp,z = flip2(y,pc,pl)
    red = np.where(z==1)[0]
    green = np.where(z==0)[0]
    tsh = 0.5
    r = get_red_th(x[red,:],.5)
    g = get_green_th(x[green,:],.5)
    chR = cp.deepcopy(x[red,:])
    chR[chR > tsh] = 1
    chG = cp.deepcopy(x[red,:])
    chG[chG > tsh] = 0
    chB = cp.deepcopy(x[red,:])
    chB[chB > tsh] = 0
    r = np.concatenate((chR, chG), axis=3)
    chR1= cp.deepcopy(x[green,:])
    chR1[chR1 > tsh] = 0
    chG1= cp.deepcopy(x[green,:])
    chG1[chG1 > tsh] = 1
    chB1= cp.deepcopy(x[green,:])
    chB1[chB1 > tsh] = 0
    g= np.concatenate((chR1, chG1), axis=3)    
    dataset =np.concatenate((r,g),axis=0)
    labels = np.concatenate((yp[red,:],yp[green,:]),axis=0)
    x_in,y_in,e_in = dataset,labels,np.ones((num_samples,1))*env
    return x_in,y_in,e_in        
    
from functools import partial

def create_environment_factory(_type_corr=None):
    """
    Possible values: "a00","a01","a10","a11","b00","b01","b10","b11","original","d00","d01","d10","d11"
    Return the environment creation function
    """
    _all_type_corr = ["a00","a01","a10","a11","b00","b01","b10","b11","original","d00","d01","d10","d11"]
    if _type_corr is None: return _all_type_corr
    if _type_corr=="a00": return partial(create_environment_a,flip=False,zero=False)
    if _type_corr=="a01": return partial(create_environment_a,flip=False,zero=True)
    if _type_corr=="a10": return partial(create_environment_a,flip=False,zero=False)
    if _type_corr=="a11": return partial(create_environment_a,flip=False,zero=True)
    if _type_corr=="b00": return partial(create_environment_b,flip=False,zero=False)
    if _type_corr=="b01": return partial(create_environment_b,flip=False,zero=True)
    if _type_corr=="b10": return partial(create_environment_b,flip=True,zero=False)
    if _type_corr=="b11": return partial(create_environment_b,flip=True,zero=True)
    if _type_corr=="original": return create_environment_original
    if _type_corr=="d00": return partial(create_environment_d,flip=False,env_only=False)
    if _type_corr=="d10": return partial(create_environment_d,flip=True,env_only=False)
    if _type_corr=="d01": return partial(create_environment_d,flip=False,env_only=True)
    if _type_corr=="d11": return partial(create_environment_d,flip=True,env_only=True)
    print("[create_environment_factory]: _type_corr={} not implemented".format(_type_corr))

#####################
#Help function to save/load dataset

def ds_save(Ds,fname,names):
    data = {name: getattr(Ds, name) for name in names }
    joblib.dump(data, fname)
def ds_load(Ds,fname):
    data = joblib.load(fname)
    for k,v in data.items(): getattr(Ds, name, v)    
    
class EnvData(object):
    def __init__(self):
        super(EnvData, self).__init__()    
        self.names=[]
    def save(self,fname):
        ds_save(self,fname,Ds.names)
    def load(self,fname):
        ds_load(Ds,fname)

    
#####################
#Main class
    
class assemble_data_mnist_any(EnvData):
    ds_names = ["MNIST","FashionMNIST","KMNIST","EMNIST","QMNIST"]
    ds_splits_EMNIST = ["byclass", "bymerge", "balanced", "letters", "digits","mnist"] # used in EMNIST
    def __init__(self, _dataset,_split, _type_corr, ntrain=None):
        super(assemble_data_mnist_any, self).__init__() 
        self._type_corr = _type_corr
        self._dataset = _dataset
        self._split=_split
        self.ntrain=ntrain
        self.names=["data_tuple_list","data_tuple_test","x_train","y_train","x_test","y_test","_type_corr"]
        self._create_env = create_environment_factory(_type_corr)
        
        if not hasattr(torchvision.datasets,_dataset):
            print("{} not present in torchvision.datasets".format(_dataset))
        
        rotateflag=False
        ds = getattr(torchvision.datasets, _dataset)
        if _dataset=="EMNIST":
            rotateflag=True
            ds_train = ds("../data/",download=True,train=True,split=_split)
            ds_test = ds("../data/",download=True,train=False,split=_split)
        else:
            ds_train = ds("../data/",download=True,train=True)
            ds_test = ds("../data/",download=True,train=False)

        x_train=ds_train.data.numpy().astype(float)
        x_test=ds_test.data.numpy().astype(float)  
        if rotateflag:
            x_train=x_train.swapaxes(-2,-1)
            x_test=x_test.swapaxes(-2,-1)
        y_train = ds_train.targets.numpy().astype(float)
        y_test = ds_test.targets.numpy().astype(float)
        if _dataset=="QMNIST":
            y_train=y_train[:,0]
            y_test=y_test[:,0]
        x_train/=float(255)
        x_test/=float(255)        
        if not self.ntrain is None: 
            x_train, y_train =  shuffle(x_train,y_train)
            x_train = x_train[:self.ntrain]
            y_train = y_train[:self.ntrain]
        num_train=x_train.shape[0]
        self.num_train=num_train
        num_test=x_test.shape[0]
        self.num_test=num_test
        
        self.x_train=x_train.reshape((num_train,28,28,1))
        self.y_train=y_train.reshape((num_train,1))
        self.x_test=x_test.reshape((num_test,28,28,1))
        self.y_test=y_test.reshape((num_test,1))
    
    def create_training_data(self, n_e, pcs, pls, seed=None, ntrain=None):
            if not seed is None: 
    #             set_seed_np(seed)
                np.random.seed(seed)
            _x = self.x_train
            _y = self.y_train
            nsamples = self.num_train
            if not ntrain is None: nsamples = min(nsamples,n_e*ntrain)
            ind_X = range(0,nsamples)
            kf = KFold(n_splits=n_e, shuffle=True)
            data_tuple_list = []
            for ke,(train, test) in enumerate(kf.split(ind_X)):
                data_tuple_list.append(self._create_env(ke,_x[test,:,:,:],_y[test,:],pcs[ke],pls[ke]))
            self.data_tuple_list = data_tuple_list
            
            (num_examples_environment,length, width, height) = self.data_tuple_list[0][0].shape # attributes of the data
            num_classes = len(np.unique(self.data_tuple_list[0][1])) # number of classes in the data    
            self.num_classes,self.num_examples_environment,self.length, self.width, self.height = num_classes, num_examples_environment, length, width, height
            
    def create_testing_data(self, corr_test, prob_label, n_e):
        x_test_mnist = self.x_test
        y_test_mnist = self.y_test
        (x_test,y_test,e_test)=self._create_env(n_e+1,x_test_mnist,y_test_mnist,corr_test,prob_label)
        self.data_tuple_test = (x_test,y_test, e_test)
    
    
import numpy as np
def save_genes(genes,labels,fname):
    print(f"writing gene data into {fname}")
    with open(fname, 'wb') as f:
        np.save(f, genes)
        np.save(f, labels)
def load_genes(fname):
    print(f"reading gene data from {fname}")
    with open(fname, 'rb') as f:
        genes = np.load(f)
        labels = np.load(f)    
        return genes,labels    
from sklearn import preprocessing
def log_min_maxs(xs):
    # Bring in log space
    xs = [np.log2(x + 1) for x in xs]
    # Normalize data
    mms = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    mms.fit(xs[0].T)    
    # it scales features so transpose is needed
    xs = [mms.transform(x.T).T for x in xs]       
    return xs
def load_data_env(n_e, path):
    xs,ys,es = [],[],[]
    for _ in range(n_e+1):
        fname = path+f"ds{_}_10k.npy"
        x,y = load_genes(fname)
        e = np.ones([x.shape[0],1])*_
        xs+=[x]
        ys+=[y]
        es+=[e]
    xs = log_min_maxs(xs)
    data_train = [ [xs[_],ys[_],es[_]]  for _ in range(n_e)]        
    data_test = [xs[n_e],ys[n_e],es[n_e]]
    return data_train, data_test   
class deconv_env(EnvData):
    """
    path = "./deconv/"
    data = deconv_env(path,2)
    data.create_training_test_data()
    data.data_tuple_list
    data.data_tuple_test
    """
    def __init__(self, path, n_e):
        super(deconv_env, self).__init__() 
        self.n_e = n_e
        self._data_train,self._data_test = load_data_env(n_e, path)    
    def create_training_test_data(self, ntrain=None):
        _reshape = lambda x: x.reshape([*x.shape,1,1])
        if ntrain: 
            self.data_tuple_list = [ [_reshape(x[:ntrain]),y[:ntrain],e[:ntrain]] for x,y,e in self._data_train]
        else:
            self.data_tuple_list = [ [_reshape(x),y,e] for x,y,e in self._data_train]

        x,y,e = self._data_test
        self.data_tuple_test =  [_reshape(x),y,e]

        num_samples, length, width, height = self.data_tuple_list[0][0].shape 
        num_classes = self.data_tuple_list[0][1].shape[1]
        self.num_classes, self.num_samples, self.length, self.width, self.height  = num_classes, num_samples, length, width, height
        return  num_classes, num_samples, length, width, height    
    