#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: irmutils.py
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


import argparse
import IPython.display as display
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import copy as cp
from sklearn.model_selection import KFold

from scipy import array, linalg, dot
from scipy.linalg import logm,  norm, expm, cholesky, inv, svd, pinv
from scipy.linalg import eig,logm,det,inv,svd,qr,pinv
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import coo_matrix, block_diag, diags, eye, kron
from scipy import sparse, stats
from scipy.sparse import linalg
from numpy import matmul, trace, diag, real, ones, eye,log
import scipy as sp
import numpy as np
import scipy.sparse

import sys, os
import time
from datetime import datetime
import itertools

from tqdm import tqdm
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
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

from GIB.data_construct import * ## contains functions for constructing data 
from GIB.IRM_methods_pt import *
from GIB.irmlib import *
from GIB.CL import *

#####################

def get_models_regr(D, args, device=None):    
    n_e, _type, dropp,  last_is_feature,n_memories,reg, hidden_dim, device = args.n_e, args._type, args.dropp, args.last_is_feature, args.n_memories, args.reg, args.hidden_dim, device
    if device is None: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes,num_samples,length, width, height = D.num_classes,D.num_samples,D.length, D.width, D.height
    model_list = [] 
    _type_list=["std","ewc","gem","mer","gated"]
    assert(_type in _type_list)    
    
    replay_batch_size, batches_per_example, beta, gamma = 5, 2, 0.03, 1.0
    if _type=="ewc":
        return EWCMLP(length, width, height, hidden_dim,num_classes,dropp=dropp,n_memories=n_memories,reg=reg,regression=True).to(device)
    if _type=="gem":
        return GEMMLP(length,width,height, hidden_dim,num_classes,dropp,n_memories,n_e,regression=True).to(device)
    if _type=="mer":
        return MERMLP(length,width,height, hidden_dim,num_classes,dropp,n_memories,n_e,replay_batch_size, batches_per_example, beta, gamma,regression=True).to(device)
    
    if last_is_feature:
        for e in range(n_e):
            if _type=="std": model_list.append(StdMLP_class(length,width,height, hidden_dim,num_classes,dropp=dropp,regression=True).to(device))
            if _type=="gated": model_list.append(GatedMLP_class(length,width,height, hidden_dim,num_classes,dropp=dropp,mask_type=args.mask_type,regression=True, entropy_type=args.entropy_type).to(device))
            if _type=="std": model_list.append(StdMLP_feature(length,width,height, hidden_dim,num_classes,dropp=dropp,regression=True).to(device))
    else:
        for e in range(n_e):            
            if _type=="std": model_list.append(StdMLP(length,width,height, hidden_dim,num_classes,dropp=dropp,regression=True).to(device))
            if _type=="gated": model_list.append(GatedMLP(length,width,height, hidden_dim,num_classes,dropp=dropp,mask_type=args.mask_type,regression=True,entropy_type=args.entropy_type).to(device))
            
    clip_value = 10.
    for model in model_list:
        for p in model.parameters():
            try:
                p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))            
            except:
                pass
    return  model_list    

def get_models_class(D, args):
    n_e, _type, dropp,  last_is_feature,n_memories,reg, hidden_dim, device = args.n_e, args._type, args.dropp, args.last_is_feature, args.n_memories, args.reg, args.hidden_dim, None
    
    if device is None: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes,num_examples_environment,length, width, height = D.num_classes,D.num_examples_environment,D.length, D.width, D.height
    model_list = [] 
    _type_list=["std","ewc","gem","mer","gated"]
    
    assert(_type in _type_list)
    replay_batch_size, batches_per_example, beta, gamma = 5, 2, 0.03, 1.0
    if _type=="ewc":
        return EWCMLP(length,width,height, hidden_dim,num_classes,dropp=dropp,n_memories=n_memories,reg=reg).to(device)
    if _type=="gem":
        return GEMMLP(length,width,height, hidden_dim,num_classes,dropp,n_memories,n_e).to(device)
    if _type=="mer":
        return MERMLP(length,width,height, hidden_dim,num_classes,dropp,n_memories,n_e,replay_batch_size, batches_per_example, beta, gamma).to(device)
    
    if last_is_feature:
        for e in range(n_e):
            if _type=="std": model_list.append(StdMLP_class(length,width,height, hidden_dim,num_classes,dropp=dropp).to(device))
            if _type=="gated": model_list.append(GatedMLP_class(length,width,height, hidden_dim,num_classes,dropp=dropp,mask_type=args.mask_type,entropy_type=args.entropy_type).to(device))
            if _type=="std": model_list.append(StdMLP_feature(length,width,height, hidden_dim,num_classes,dropp=dropp).to(device))
    else:
        for e in range(n_e):            
            if _type=="std": model_list.append(StdMLP(length,width,height, hidden_dim,num_classes,dropp=dropp).to(device))
            if _type=="gated": model_list.append(GatedMLP(length,width,height, hidden_dim,num_classes,dropp=dropp,mask_type=args.mask_type,entropy_type=args.entropy_type).to(device))            
    clip_value = 10.
    for model in model_list:
        for p in model.parameters():
            try:
                p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))            
            except:
                pass
    return  model_list



#####################
import random
def set_seed_np(seed):
    np.random.seed(seed)
def set_seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.cuda.manual_seed_all(seed)    

def get_data_args(args):
    if args._type_data.upper() in ['DECONV']:
        return get_data_deconv(args)
    else:
        return get_data(args.n_e, args._type_data,args._type_split, [args.pc_train_start,args.pc_train_end] ,args.pl_train,args.pc_test, args.pl_test, args._type_corr, args.seed, args.ntrain)

def get_data(n_e, _type_data,_type_split, pc_train,pl_train,pc_test, pl_test, _type_corr,seed, ntrain=None):
    if False:
        pc_train, pl_train = [.2,.1], .25
        pc_test, pl_test = 0.9, .25
    
      # number of environments
    p_label_list = [pl_train]*n_e # list of probabilities of switching pre-label
    
    if type(pc_train) == float: 
        p_color_list = [pl_train]*n_e
    elif len(pc_train)==n_e:
        p_color_list = pc_train
    else:
        p_color_list = np.linspace(pc_train[0],pc_train[1],n_e) # list of probabilities of switching the final label to obtain the color index

    p_label_test = pl_test # probability of switching pre-label in test environment
    p_color_test = pc_test  # probability of switching the final label to obtain the color index in test environment
        
    D = assemble_data_mnist_any(_type_data,_type_split,_type_corr)
    
    D.create_training_data(n_e, p_color_list, p_label_list,seed, ntrain) # creates the training environments
    D.create_testing_data(p_color_test, p_label_test, n_e)  # sets up the testing environment
    
    (num_examples_environment,length, width, height) = D.data_tuple_list[0][0].shape # attributes of the data
    num_classes = len(np.unique(D.data_tuple_list[0][1])) # number of classes in the data    
    D.num_classes,D.num_examples_environment,D.length, D.width, D.height = num_classes,num_examples_environment,length, width, height
    return D,num_classes,num_examples_environment,length, width, height

def get_data_deconv(args):
    path = "../deconv/"
    D = deconv_env(path,args.n_e)
    D.create_training_test_data(args.ntrain)
    return D,D.num_classes,D.num_samples, D.length, D.width, D.height

from dblog.dblog import DBlog
#####################
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
def runit_args(D,args):
    start = time.time()
    
    dblog = DBlog()
    dblog.connect()
    try: dblog.create_table('gib_configs')
    except: pass
    try: dblog.create_table('gib_runs')
    except: pass
    dblog.insert(dict(vars(args)), 'gib_configs')

    logger = ExperimentWriter(f"logs/{args.rnd}")
    logger.log_hparams(dict(vars(args)))
    
    
    writer = None
    _now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.writer_flag:        
        writer = SummaryWriter('runs/{}_{}_{}_{}_rnd_{}'.format(args.ver,_now,args._type_solver,args._type,args.rnd))
    
    args.writer = writer
    args.logger = logger
    args.dblog = dblog    
    
    if args.regression:
        model_list = get_models_regr(D, args, device=None)
    else:
        model_list = get_models_class(D, args)

    if args._type_solver != "irmg":
        if type(model_list)==list:
            model_list = model_list[0]

    train_factory(args)(D.data_tuple_list,D.data_tuple_test, model_list, args, writer)        
    stat_evaluate_args(D, model_list, args)           
    _mean_acc,_std_acc = stat_evaluate_args(D, model_list, args)
    if args.regression:
        print (f"Training rmse {_mean_acc[0]}") 
        print (f"Training pcorr {_mean_acc[1]}") 
        print (f"Testing rmse {_mean_acc[2]}")
        print (f"Testing pcorr {_mean_acc[3]}")
    else:
        print (f"\nTraining accuracy {100*_mean_acc[0]}") 
        print (f"Testing accuracy {100*_mean_acc[1]}")
    elapsed = (time.time() - start)
    if writer: writer.flush()
    if writer: writer.close()
    save_perf(_mean_acc,_std_acc,elapsed, args)
    return model_list, _mean_acc,_std_acc,elapsed


#####################
from sklearn.utils import shuffle
def plot_images(x_in,y_in,e_in,nsamples=10,shuffleflag=False):
    if shuffleflag: x_in, y_in,e_in = shuffle(x_in, y_in,e_in)
    nplot = min(nsamples,x_in.shape[0])
#     plt.axis(False)
    fig, axs = plt.subplots(nplot,2,figsize=(8,4*nplot))
    # axs = axs.flatten()
    for ki in range(nplot):
        axs[ki,0].set_title(y_in[ki])
        axs[ki,0].imshow(x_in[ki][:,:,0])
#         axs[ki,1].set_title("yp,z={},{}".format(yp[ki],z[ki]))
        axs[ki,1].imshow(x_in[ki][:,:,1])
        axs[ki,0].axis(False)
        axs[ki,1].axis(False)
    plt.axis(False)
#####################


def add_channel(images,islast=True):
    fn = np.zeros
    if islast:
        _ch = fn([*images[:,:,:,0].shape,1])
        return np.concatenate([images,_ch],axis=-1)
    else:
        lst = list(images[:,0,:,:].shape)
        _shape = [lst[0]]+[1]+lst[1:]
        _ch = fn(_shape)
        return np.concatenate([images,_ch],axis=1)
    
# functions to show an image
def imshow(img,title,ax=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if title: plt.title(title)
    plt.axis(False)
    cmap = plt.cm.spring 
    if ax is None:
        return plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
    else:
        ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
#     plt.show()

def plot_samples_env(D,e,nsamples=10, usetest=False,shuffleflag=True, title=None):
    if usetest:
        images, labels = D.data_tuple_test[0],D.data_tuple_test[1]
    else:
        images, labels = D.data_tuple_list[e][0],D.data_tuple_list[e][1]
    
    if shuffleflag: images, labels = shuffle(images, labels)
    images, labels = images[:nsamples],labels[:nsamples]
    
    # show images
    images = add_channel(images)
    images = images.swapaxes(3,1)
    images = images.swapaxes(2,3)
    images_pt = torchvision.utils.make_grid(torch.Tensor(images)).cpu()
    if title is None:
        title = ' '.join('{}' .format(int(labels[j])) for j in range(len(images)))
    return imshow(images_pt,title)

def plot_samples(x_in,y_in,e_in,nsamples=10,shuffleflag=True, title=None, titleflag=False):
    images, labels = x_in,y_in
    if shuffleflag: images, labels = shuffle(images, labels)
    images, labels = images[:nsamples],labels[:nsamples]
    
    # images, labels = torch.Tensor(images).to(device), torch.Tensor(labels).to(device)
    # images = images.swapaxes(3,1)
    # show images
    images = add_channel(images)
    images = images.swapaxes(3,1)
    images = images.swapaxes(2,3)
    images_pt = torchvision.utils.make_grid(torch.Tensor(images)).cpu()
    if not title: 
        title = ' '.join('{}' .format(int(labels[j])) for j in range(len(images)))
    if not titleflag: title = None
    return imshow(images_pt,title)
    
    
#####################
def stat_evaluate_args(D, model, args):
    train_accs,test_accs = [],[]
    train_rmses,train_pcors,test_rmses,test_pcors =[],[],[],[]
    for _ in range(args.eval_ntimes):
        if args.regression:
            (test_rmse, test_pcor), (train_rmse, train_pcor) = evaluate_factory(args)(D.data_tuple_test, D.data_tuple_list, model)
            train_rmses+=[train_rmse]
            train_pcors+=[train_pcor]
            test_rmses+=[test_rmse]
            test_pcors+=[test_pcor]
        else:
            test_acc,train_acc = evaluate_factory(args)(D.data_tuple_test, D.data_tuple_list, model)
            train_accs+=[train_acc]
            test_accs+=[test_acc]

    if args.regression:
        perf = np.array([train_rmses,train_pcors,test_rmses,test_pcors])
    else:
        perf = np.array([train_accs,test_accs])
    _mean = np.mean(perf.T,axis=0)
    _std = np.std(perf.T,axis=0)
    if args.verbose:
        print("train,test (mean) acc = {}".format(_mean))
        print("train,test (std) acc = {}".format(_std))
    return _mean,_std

#####################
import itertools
def get_params_str(args, sep=", ",sepkv=", "):
    return "{}".format(sep).join(["{}{}{}".format(name,sepkv, value ) for name, value in args._get_kwargs()  if value is not None ])
def _merge(names, values): return {n:v for n,v in zip(names, values)}
def _merge(names, values): return ' '.join(["--{} {}".format(n,v) for n,v in zip(names, values)]+[""])
def _iter(names, lstlst):    
    for _ in itertools.product(*lstlst): 
        yield _merge(names,_)   
def tobool(v): 
    if v.lower() in ['true']: return True
    elif v.lower() in ['false']: return False
    else: raise ValueError("invalid literal for tobool(): {}".format(v))
def myconvert(v):
    try: 
        return int(v)
    except:
        try: 
            return float(v)
        except:
            try: 
                return tobool(v)
            except:
                return v
#####################        
def save_perf(_mean_acc,_std_acc,elapsed, args):        
    with open("{}/perf_{}.txt".format(args.output_path,args.ver),"a+t") as fp:
        str_params = get_params_str(args)  
        if args.regression:
            names = "train_rmse,train_pcorr,test_rmse,test_pcorr, train_rmse_std,train_pcorr_std,test_rmse_std,test_pcorr_std,elapsed".split(",")
            acc_str = ",".join("{},{}".format(n,v) for n,v in zip(names,[*_mean_acc,*_std_acc,elapsed]))
        else:
            names = "train_acc, test_acc, train_acc_std, test_acc_std ,elapsed".split(",")
            acc_str = ",".join("{},{}".format(n,v) for n,v in zip(names,[*_mean_acc,*_std_acc,elapsed]))            
        fp.write(str_params +","+ acc_str+"\n")
#####################        
        

