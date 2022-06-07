#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: main_sim.py
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
from sklearn.utils import shuffle
import pandas as pd
import copy as cp
from sklearn.model_selection import KFold

# from scipy import array, linalg, dot
# from scipy.linalg import logm,  norm, expm, cholesky, inv, svd, pinv
# from scipy.linalg import eig,logm,det,inv,svd,qr,pinv
# from scipy.sparse.linalg import spsolve, cg
# from scipy.sparse import coo_matrix, block_diag, diags, eye, kron
# from scipy import sparse, stats
# from scipy.sparse import linalg
# from numpy import matmul, trace, diag, real, ones, eye,log
import scipy as sp
import numpy as np
import scipy.sparse

import sys, os
import matplotlib.pyplot as plt


import time
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



import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import importlib
import argparse
import random
import uuid
import time
import os


import itertools

from GIB.data_construct import * ## contains functions for constructing data 
from GIB.IRM_methods_pt import *
from GIB.irmlib import *
from GIB.irmutils import *

from random import randint

from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

#################
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_parser():
    parser = argparse.ArgumentParser(description='Continual IRM')

    # experiment parameters
    parser.add_argument('--seed', type=int, default=1234,     help='random seed of model')
    parser.add_argument('--output_path', type=str, default='results/', help='save models at the end of training')
    
    parser.add_argument('--writer_flag', type=str2bool, nargs='?', const=True, default=True, help='writer_flag')
    parser.add_argument('--scheduler_flag', type=str2bool, nargs='?', const=True, default=False, help='scheduler_flag')
    parser.add_argument('--scheduler_type', type=str, default='plateau', help="'plateau','exponential','cyclic'")
    
    parser.add_argument('--ver', type=str, default="{}".format(np.random.randint(1e6)), help='ver')
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help='verbose')
    
 
    #this model
    parser.add_argument('--n_e', type=int, default=2, help='n_e')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs per env')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--ntrain', type=int, default=None, help='number of sample to train')
    parser.add_argument('--eval_ntimes', type=int, default=10, help='eval_ntimes')
    

    parser.add_argument('--_type_corr', type=str, default='b11', help='_type_corr')
    parser.add_argument('--_type_data', type=str, default='MNIST', help='_type_data')
    parser.add_argument('--_type_split', type=str, default='letters', help='_type_split')    
    
    parser.add_argument('--weight_decay', type=float, default=.00125, help='weight_decay')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='learning_rate')
    parser.add_argument('--dropp', type=float, default=0, help='dropps')
    parser.add_argument('--batch_size_step', type=int, default=4, help='batch_size_step')
    
    parser.add_argument('--entropy_type', type=str, default="shannon", help='shannon, renyi')
    
    #CL
    parser.add_argument('--replay_batch_size', type=int, default=5, help='replay_batch_size size')
    parser.add_argument('--batches_per_example', type=int, default=2, help='batches_per_example size')
    parser.add_argument('--beta', type=float, default=0.03, help='beta')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma')

    #Dataset
    parser.add_argument('--pc_train_start', type=float, default=.2, help='pc_train_start')
    parser.add_argument('--pc_train_end', type=float, default=.1, help='pc_train_end')
    parser.add_argument('--pl_train', type=float, default=.25, help='pl_train')
    parser.add_argument('--pc_test', type=float, default=.9, help='pc_test')
    parser.add_argument('--pl_test', type=float, default=.25, help='pl_test')    
    
    #Neural Network Model 
    parser.add_argument('--_type', type=str, default="std", help='_type')
    parser.add_argument('--hidden_dim', type=int, default=390, help='hidden_dim')

    parser.add_argument('--regression', type=str2bool, nargs='?', const=True, default=False, help='regression')        
    #other
    parser.add_argument('--last_is_feature', type=str2bool, nargs='?', const=True, default=False, help='last_is_feature')
    # parser.add_argument('--sequential_flag', type=str2bool, nargs='?', const=True, default=True, help='sequential_flag')
    parser.add_argument('--training_type', type=str, default="parallel", help='training_type (sequential,parallel,one)')

    parser.add_argument('--parallel_every_epoch_flag', type=str2bool, nargs='?', const=True, default=False, help='parallel_every_epoch_flag, aletrnative is every batch')
    
    #Algorith
    parser.add_argument('--_type_solver', type=str, default="admm", help='_type_solver')

    #GEM
    #EWC
    parser.add_argument('--n_memories', type=int, default=3, help='n_memories')
    parser.add_argument('--reg', type=float, default=100, help='reg')
    
    #IRM
    parser.add_argument('--gamma_new', type=float, default=91257, help='gamma_new')
    parser.add_argument('--steps_threshold', type=int, default=3, help='steps_threshold')
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.00110794568, help='l2_regularizer_weight')
    

    #IRMG
    parser.add_argument('--termination_acc', type=float, default=.6, help='termination_acc')
    parser.add_argument('--warm_start', type=int, default=300, help='warm_start')    
    
    #PERM
    parser.add_argument('--use_gate_net_flag', type=str2bool, nargs='?', const=True, default=True, help='use_gate_net_flag')
    parser.add_argument('--lambda0', type=float, default=1e-1, help='lambda0')
    parser.add_argument('--lambda1', type=float, default=1e1, help='lambda1')
    parser.add_argument('--lambda0before', type=float, default=1, help='lambda0before')
    parser.add_argument('--lambda1before', type=float, default=1, help='lambda1before')
    
    parser.add_argument('--mask_type', type=str, default="discrete", help='tanh, sigmoid, discrete')
    
    parser.add_argument('--mask_update_every', type=int, default=100, help='mask_update_every')    
    parser.add_argument('--net_freeze_epoch', type=int, default=500, help='net_freeze_epoch')    
    
    parser.add_argument('--rnd', type=str, default="{:0d}".format(randint(0,1e6)))
    
    return parser
        
if __name__ == "__main__":
    parser = get_parser()
    
    args = parser.parse_args()
    
    args.now = f"{datetime.now():%Y-%m-%d_%H:%M:%S}"
    print(dict(vars(args)))    

    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds
    set_seed_all(args.seed)
        
    if torch.cuda.is_available(): torch.multiprocessing.set_start_method('spawn')
    if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    
    if args.last_is_feature and args._type_solver!="irmg": 
        print("parameters not valid") 
        exit
    if (args._type=="ewc" and args._type_solver!="ewc" ) or (args._type!="ewc" and args._type_solver=="ewc"): 
        print("parameters not valid") 
        exit 
    
    args.check_loop_fn = None
    D,num_classes,num_examples_environment,length, width, height = get_data_args(args)
    
    model_list, _mean_acc,_std_acc,elapsed = runit_args(D,args)
    print("simulation ended")    