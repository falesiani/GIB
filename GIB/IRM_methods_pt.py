#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: IRM_methods_pt.py
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
from torchvision import transforms as T
from torch.optim import Adam, SGD, lr_scheduler

import sys
import copy
import math

from GIB.utils import *
# from pytorch_lightning.loggers.csv_logs import ExperimentWriter

def get_scheduler(optimizer,args):
    scheduler = None
    if args.scheduler_type in ['plateau']: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=False)
    if args.scheduler_type in ['exponential']: scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    if args.scheduler_type in ['cyclic']: scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001,cycle_momentum=False)
    return scheduler

def get_optimizer(model,args):
    opt_cls = optim.Adam
#         opt_cls = optim.SGD
    optimizer = opt_cls(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return optimizer

def get_optimizer_classifier(model,args):
    opt_cls = optim.Adam
#         opt_cls = optim.SGD
    optimizer = opt_cls(model.get_classifier_net().parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return optimizer

def get_optimizer_gate(model,args):
    opt_cls = optim.Adam
#         opt_cls = optim.SGD
    optimizer = opt_cls(model.get_gate_net().parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return optimizer

def get_optimizer_list_fix(model_list,args):
    optimizer_list = []
    n_e = len(model_list)
    opt_cls = optim.Adam
#         opt_cls = optim.SGD
    for e in range(n_e):            
        optimizer_list.append(opt_cls(model_list[e].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay))
    return optimizer_list

def get_optimizer_list_variable(model_list,args):
    optimizer_list = []
    for e in range(n_e+1):      
        opt_cls = optim.Adam
#             opt_cls = optim.SGD
        if (e<=n_e-1):
            optimizer_list.append(opt_cls(model_list[e].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay))
        if (e==n_e):
            optimizer_list.append(opt_cls(model_list[e].parameters(), lr=args.learning_rate*0.1, weight_decay=args.weight_decay))
    return optimizer_list

def get_optimizer_factory(args):
    if args.last_is_feature is False:
        return get_optimizer_list_fix
    else:
        return get_optimizer_list_variable

    
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
                        
def train_sequential(data_tuple_list, data_tuple_test, model_list, args, writer=None):
        n_e  = len(data_tuple_list) 
        
        x_in,y_in,e_in = concat_envs(data_tuple_list)           
        y_in = torch.Tensor(y_in)
        x_in = torch.Tensor(x_in)
        
        x_in_te = data_tuple_test[0]
        y_in_te = data_tuple_test[1]
        y_in_te = torch.Tensor(y_in_te)
        x_in_te = torch.Tensor(x_in_te)   

        set_models(model_list,train_flag=True)

        # initialize optimizer for all the environments and representation learner and store it in a list
        if args._type_solver.upper() in ['IRMG']:
            optimizer_list = get_optimizer_factory(args)(model_list,args)
            model = model_list[0]
        else:    
            if type(model_list)==list:
                model = model_list[0]
            else:
                model = model_list
            optimizer = get_optimizer(model,args)
            if args._type_solver.upper() in ['PIRM',"PERM"]:                
                optimizer_gate = get_optimizer_gate(model,args)
                optimizer_classifier = get_optimizer_classifier(model,args)
            scheduler = None
            if args.scheduler_flag :
                scheduler = get_scheduler(optimizer,args)
                
        
        ####### train
        num_examples = data_tuple_list[0][0].shape[0]
        global_iter           = 0
        w = torch.tensor(1.).cuda().requires_grad_() 
        first_time = True
        for kenv in range(n_e):
            print(f"({kenv})",end="")
            et = torch.Tensor(kenv)  
            if args.verbose: print ("Env: {}/{}".format(kenv,n_e))
            else: print("|",end="")
                        
            env_iter = 0
            for epoch in range(args.num_epochs):  
                if epoch> 0 and not args.check_loop_fn is None:
                    if args.check_loop_fn(args,score,epoch):
                        return 'stopped'                
                if args.verbose: print ("Epoch: "  + str(epoch)+", {}/{}".format(kenv,n_e))
                else: print(".",end="")
                datat_list = shuffle_envs(data_tuple_list)
                
                for count, offset in enumerate(range(0,num_examples, args.batch_size)): 
                    end = offset + args.batch_size
                    env_iter +=1
                    global_iter +=1
                    if args.verbose: print("{}/{}".format(kenv,n_e),end='|')
                    x = datat_list[kenv][0][offset:end,:]
                    y = datat_list[kenv][1][offset:end,:]   
                    x = torch.Tensor(x)
                    y = torch.Tensor(y) 
                                        
                    assert(args._type_solver.upper() in ["ERM","IRM","PIRM","IRMG","PERM","CL"])                
                    assert(int(kenv)==kenv)
                    
                    #ERM
                    if args._type_solver.upper()=="ERM":                    
                        y_ = model(x)
                        if model.type_()=="std":
                            loss_value = model.eval_loss(y_,y)
                            if writer and epoch%10==0: writer.add_scalar("Loss0-{}/train".format(kenv) ,loss_value, env_iter)
                        else:
                            loss_value,loss0,loss1,loss2 = model.eval_loss_debug(y_,y)
                            if writer and epoch%10==0: writer.add_scalar("Loss0-{}/train".format(kenv) ,loss0, env_iter)
                            if writer and epoch%10==0: writer.add_scalar("Loss1-{}/train".format(kenv) ,loss1, env_iter)
                            if writer and epoch%10==0: writer.add_scalar("Loss2-{}/train".format(kenv) ,loss2, env_iter)
                            if writer and epoch%10==0: writer.add_scalar("Energy-{}/train".format(kenv) , model.get_energy(), env_iter)                           

                        optimizer.zero_grad()
                        loss_value.backward()
                        optimizer.step()

                        y_ = model(x_in)
                        if args.regression:
                            rmse_train,pcor_train = metric_regression(y_,y_in)
                        else:            
                            acc_train = np.float(eval_accuracy(y_,y_in))                
                    
                    #IRMv1
                    if args._type_solver.upper()=="IRM":
                        gamma     = 1.0
                        if(epoch>=args.steps_threshold):
                            gamma = args.gamma_new   
                        loss_value, loss0, loss_penalty  = loss_total_debug(model,x,et,y,w,gamma,n_e)
                        optimizer.zero_grad()
                        loss_value.backward()
                        optimizer.step()            


                        if writer and epoch%10==0: writer.add_scalar("Loss0-{}/train".format(kenv) ,loss0, env_iter)
                        if writer and epoch%10==0: writer.add_scalar("Loss_penalty-{}/train".format(kenv) ,loss_penalty, env_iter)                         

                        y_ = model(x_in)
                        if args.regression:
                            rmse_train,pcor_train = metric_regression(y_,y_in)
                        else:
                            acc_train = np.float(eval_accuracy(y_,y_in))

                        
                    #PERM
                    if args._type_solver.upper()=="PERM":     
                        if False: print(f"{kenv}",end="/")
                            
                                                    
                        if kenv==0 and epoch<args.net_freeze_epoch:
                            y_ = model(x)
                            loss0, loss1 = model.eval_loss(y_,y), model.get_entropy_loss(x)

                            if writer and epoch%10==0: writer.add_scalar("Loss0-{}/train".format(kenv) ,loss0, env_iter)
                            if writer and epoch%10==0: writer.add_scalar("Loss1-{}/train".format(kenv) ,loss1, env_iter)
                            loss_value = args.lambda0before*loss0  + args.lambda1before*loss1

                            optimizer.zero_grad()
                            loss_value.backward()
                            optimizer.step()

                        else:
                            y_ = model(x)
                            loss0, loss1 = model.eval_loss(y_,y), model.get_gated_entropy_loss(x)

                            if writer and epoch%10==0: writer.add_scalar("Loss0-{}/train".format(kenv) ,loss0, env_iter)
                            if writer and epoch%10==0: writer.add_scalar("Loss1-{}/train".format(kenv) ,loss1, env_iter)

                            loss_value = args.lambda0*loss0  + args.lambda1*loss1

                            if args.use_gate_net_flag:
                                optimizer_gate.zero_grad()
                                loss_value.backward()
                                if False: print(f"grads = {sum([(p.grad**2).sum() for p in model.get_gate_net().parameters()])}")
                                    
                                    
                                optimizer_gate.step()
                            else:
                                optimizer_classifier.zero_grad()
                                loss_value.backward()
                                optimizer_classifier.step()

                        if epoch%args.mask_update_every==args.mask_update_every-1:
                            model.update_mask()
                            print(f"mask={(model.get_mask()>0.5).long().sum()}",end=", ")                             

                            if False:
                                y_ = model(x_in)
                                acc_train = np.float(eval_accuracy(y_,y_in))   
                        if writer and epoch%10==0: writer.add_scalar("Mask-{}/train".format(kenv) ,(model.get_mask()>0.5).long().sum(), env_iter)

                            
                    #PRJ-IRMv1
                    if args._type_solver.upper()=="PIRM":

                        if kenv==0:
                            gamma     = 0.0
                            loss_value = loss_total(model,x,et,y,w,gamma,n_e)
                            optimizer.zero_grad()
                            loss_value.backward()
                            optimizer.step()
                        else:                            
                            gamma     = 0.0
                            if(epoch>=args.steps_threshold):
                                gamma = args.gamma_new   
                            loss_value = loss_total(model,x,et,y,w,gamma,n_e)
                            
                            optimizer_gate.zero_grad()
                            loss_value.backward()
                            optimizer_gate.step()                            

                        y_ = model(x_in)
                        if args.regression:
                            rmse_train,pcor_train = metric_regression(y_,y_in)
                        else:                        
                            acc_train = np.float(eval_accuracy(y_,y_in))
                        
                        if epoch%args.mask_update_every==args.mask_update_every-1:
                            model.update_mask()
                            print(f"mask={(model.get_mask()>0.5).long().sum()}",end=", ")    
                        if writer and epoch%10==0: writer.add_scalar("Mask-{}/train".format(kenv) ,(model.get_mask()>0.5).long().sum(), env_iter)
                        
                    
                    #IRMG
                    if args._type_solver.upper()=="IRMG":
                        #IRMG output computation
                        y_ = combine_models(model_list[:(kenv+1)], x)
                        loss_value = model_list[kenv].eval_loss(y_,y)

                        
                        optimizer_list[kenv].zero_grad()
                        loss_value.backward()
                        optimizer_list[kenv].step()

                        # computing training accuracy
                        y_ = combine_models(model_list[:(kenv+1)], x_in)
                        if args.regression:
                            rmse_train,pcor_train = metric_regression(y_,y_in)
                        else:                        
                            acc_train = np.float(eval_accuracy(y_,y_in))
                        

                    #CL    
                    if args._type_solver.upper()=="CL":
                        if model.type_() in ["gem","ewc"]:
                            loss_value = model.get_cl_loss(kenv,x,y.long())
                        elif model.type_() == "mer":
                            model.step(kenv,x,y.long(), optimizer, args.batch_size_step)
                            y_ = model(x)
                            loss_value = model.eval_loss(y_,y)                        
                        else:    
                            y_ = model(x)
                            loss_value = model.eval_loss(y_,y)

                        if model.type_()!="mer":
                            optimizer.zero_grad()
                            loss_value.backward()
                            optimizer.step()

                        y_ = model(x_in)
                        if args.regression:
                            rmse_train,pcor_train = metric_regression(y_,y_in)
                        else:
                            acc_train = np.float(eval_accuracy(y_,y_in).detach())
    
                    # register performances
                    y_ = model(x_in)
                    y_env = model(x)
                    y_te_ = model(x_in_te)
                    
                    try:
                        _counter+=1
                    except:
                        _counter=0                    
                    
                    if args._type_solver.upper() in ['PIRM',"PERM"]: 
                        ngates = torch.sum(model.get_mask()>.5).detach().item()
                    else:
                        ngates = 0
                    if args.regression:
                        rmse_train_tr,pcor_train_tr = metric_regression(y_,y_in).item()
                        rmse_train_env, pcor_train_env = metric_regression(y_env,y).item()
                        rmse_train_te,pcor_train_te = metric_regression(y_te_,y_in_te).item()
                        # metrics = {
                        #     f'Loss-{kenv}/train':loss_value.detach().item()
                        #     ,f"rmse-{kenv}/train_tr" :rmse_train_tr
                        #     ,f"pcor-{kenv}/train_tr" :pcor_train_tr
                        #     ,f"rmse-{kenv}/train_env" :rmse_train_env
                        #     ,f"pcor-{kenv}/train_env" :pcor_train_env
                        #     ,f"rmse-{kenv}/train_te" :rmse_train_te
                        #     ,f"pcor-{kenv}/train_te" :pcor_train_te
                        #     ,f"ngates-{kenv}/train": ngates
                        #     ,f"kenv": kenv
                        #     ,f"env_iter": env_iter
                        # }
                        metrics = {
                            f'Loss/train':loss_value.detach().item()
                            ,f"rmse/train_tr" :rmse_train_tr
                            ,f"pcor/train_tr" :pcor_train_tr
                            ,f"rmse/train_env" :rmse_train_env
                            ,f"pcor/train_env" :pcor_train_env
                            ,f"rmse/train_te" :rmse_train_te
                            ,f"pcor/train_te" :pcor_train_te
                            ,f"ngates/train": ngates
                            ,f"kenv": kenv
                            ,f"env_iter": env_iter
                            ,"epoch":epoch
                            , 'rnd': args.rnd                            
                        }
                        
                    else:                        
                        acc_train_tr = eval_accuracy(y_,y_in).item()
                        acc_train_env = eval_accuracy(y_env,y).detach().item()
                        acc_train_te = eval_accuracy(y_te_,y_in_te).detach().item()
                        # metrics = {
                        #     f'Loss-{kenv}/train':loss_value.item()
                        #     ,f"Accuracy-{kenv}/train_tr" :acc_train_tr
                        #     ,f"Accuracy-{kenv}/train_env" :acc_train_env
                        #     ,f"Accuracy-{kenv}/train_te" :acc_train_te
                        #     ,f"ngates-{kenv}/train": ngates
                        #     ,f"kenv": kenv
                        #     ,f"env_iter": env_iter                            
                        # }  
                        metrics = {
                            f'Loss/train':loss_value.item()
                            ,f"Accuracy/train_tr" :acc_train_tr
                            ,f"Accuracy/train_env" :acc_train_env
                            ,f"Accuracy/train_te" :acc_train_te
                            ,f"ngates/train": ngates
                            ,f"kenv": kenv
                            ,f"env_iter": env_iter
                            ,"epoch":epoch
                            , 'rnd': args.rnd
                        }                          
                                     
                    # args.logger.log_metrics(metrics, step=env_iter)
                    args.logger.log_metrics(metrics, step=_counter)
                    if args.verbose:
                        print(f"hello:metrics={metrics}")
                    args.logger.save()   
                    
                    try: 
                        args.dblog.insert(metrics, 'gib_runs')
                    except: pass
                    
        
                    if writer and epoch%10==0: 
                        writer.add_scalar("Loss-{}/train".format(kenv), loss_value, env_iter)
                        if args.regression:
                            rmse_train,pcor_train = metric_regression(y_,y_in)
                            writer.add_scalar("rmse-{}/train".format(kenv) ,rmse_train, env_iter)
                            writer.add_scalar("pcor-{}/train".format(kenv) ,pcor_train, env_iter)
                            rmse_train, pcor_train = metric_regression(y_env,y)
                            writer.add_scalar("rmse-{}/train_this_env".format(kenv) ,rmse_train, env_iter)
                            writer.add_scalar("pcor-{}/train_this_env".format(kenv) ,pcor_train, env_iter)
                            rmse_train,pcor_train = metric_regression(y_te_,y_in_te)
                            writer.add_scalar("rmse-{}/test_while_training".format(kenv) ,rmse_train, env_iter)
                            writer.add_scalar("pcor-{}/test_while_training".format(kenv) ,pcor_train, env_iter)
                        else:                        
                            acc_train = np.float(eval_accuracy(y_,y_in))                   
                            writer.add_scalar("Accuracy-{}/train".format(kenv) ,acc_train, env_iter)
                            acc_train_env = np.float(eval_accuracy(y_env,y).detach())
                            writer.add_scalar("Accuracy-{}/train_this_env".format(kenv) ,acc_train_env, env_iter)
                            acc_train_te = np.float(eval_accuracy(y_te_,y_in_te).detach())
                            writer.add_scalar("Accuracy-{}/test_while_training".format(kenv) ,acc_train_te, env_iter)

        return 'finished'

def train_together(data_tuple_list,data_tuple_test, model_list, args, writer=None):
        n_e  = len(data_tuple_list) 
        x_in,y_in,e_in = concat_envs(data_tuple_list)           
        y_in = torch.Tensor(y_in)
        x_in = torch.Tensor(x_in)
        
        x_in_te = data_tuple_test[0]
        y_in_te = data_tuple_test[1]
        y_in_te = torch.Tensor(y_in_te)
        x_in_te = torch.Tensor(x_in_te)                
        
        set_models(model_list,train_flag=True)

        # initialize optimizer for all the environments and representation learner and store it in a list
        if args._type_solver.upper() in ['IRMG']:
            optimizer_list = get_optimizer_factory(args)(model_list,args)
            model = lambda _x: combine_models(model_list, _x)
        else:    
            if type(model_list)==list:
                model = model_list[0]
            else:
                model = model_list
            optimizer = get_optimizer(model,args)
            if args._type_solver.upper() in ['PIRM',"PERM","PIRM"]:                
                optimizer_gate = get_optimizer_gate(model,args)
                optimizer_classifier = get_optimizer_classifier(model,args)            
            scheduler = None
            if args.scheduler_flag :
                scheduler = get_scheduler(optimizer,args)


        ####### train
        num_examples = data_tuple_list[0][0].shape[0]
        global_iter           = 0
        w = torch.tensor(1.).cuda().requires_grad_() 
        first_time = True
        for epoch in range(args.num_epochs):    
            if epoch and not args.check_loop_fn is None:
                if args.check_loop_fn(args,score,epoch):
                    return 'stopped'
            if args.verbose: print ("Epoch: {}/{}({})".format(epoch,args.num_epochs,n_e))
            else: print(".",end="")
            datat_list = shuffle_envs(data_tuple_list)
            if args.parallel_every_epoch_flag:
                global_iter +=1
            for count, offset in enumerate(range(0,num_examples, args.batch_size)): 
                end = offset + args.batch_size
                if not args.parallel_every_epoch_flag:
                    global_iter +=1
                which_env = global_iter % n_e                
                if args.verbose: print("{}/{}".format(which_env,n_e),end='|')
                x = datat_list[which_env][0][offset:end,:]
                y = datat_list[which_env][1][offset:end,:]   
                x = torch.Tensor(x)
                y = torch.Tensor(y) 
                et = torch.Tensor(which_env)
                
                # xs = [torch.Tensor(datat_list[which_env][0][offset:end,:]) for _ in range(n_e)]
                # ys = [torch.Tensor(datat_list[which_env][1][offset:end,:]) for _ in range(n_e)]
                xs = [torch.Tensor(datat_list[_][0][offset:end,:]) for _ in range(n_e)]
                ys = [torch.Tensor(datat_list[_][1][offset:end,:]) for _ in range(n_e)]
                x_envs = torch.concat(xs)
                y_envs = torch.concat(ys)

                assert(args._type_solver.upper() in ["ERM","IRM","IRMG","CL","PERM","PIRM"])
                
                
                #ERM
                if args._type_solver.upper()=="ERM":         
                    for x_temp,y_temp in zip(xs,ys):
                        y_pred = model(x_temp)
                        loss_value = model.eval_loss(y_pred,y_temp)

                        if args._type in ['gated']:
                            loss_value += args.lambda0*model.gate3.get_loss()
                            loss_value += args.lambda1*model.get_entropy_loss(x_temp)

                    optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()

                    y_ = model(x_in)
                    if args.regression:
                        rmse_train,pcor_train = metric_regression(y_,y_in)
                    else:                                
                        acc_train = np.float(eval_accuracy(y_,y_in))
                        
                    if args._type in ['gated'] and epoch%args.mask_update_every==args.mask_update_every-1:
                        model.update_mask()
                        print(f"mask={(model.get_mask()>0.5).long().sum()}",end=", ")                         
                    
#                 #ERM
#                 if args._type_solver.upper()=="ERM":                    
#                     y_ = model(x)
#                     loss_value = model.eval_loss(y_,y)
                    
#                     if args._type in ['gated']:
#                         loss_value += args.lambda0*model.gate3.get_loss()
#                         loss_value += args.lambda1*model.get_entropy_loss(x)

#                     optimizer.zero_grad()
#                     loss_value.backward()
#                     optimizer.step()

#                     y_ = model(x_in)
#                     if args.regression:
#                         rmse_train,pcor_train = metric_regression(y_,y_in)
#                     else:                                
#                         acc_train = np.float(eval_accuracy(y_,y_in))
                        
#                     if args._type in ['gated'] and epoch%args.mask_update_every==args.mask_update_every-1:
#                         model.update_mask()
#                         print(f"mask={(model.get_mask()>0.5).long().sum()}",end=", ")                         

                #PERM (similar to irm)
                if args._type_solver.upper()=="PERM":       
                    if epoch<args.net_freeze_epoch:
                        loss_value = 0.
                        gamma = 1.
                        for x_temp,y_temp in zip(xs,ys):
                            logits = model(x_temp)    
                            logits = F.log_softmax(logits, dim=1)
                            loss_value += args.lambda0before*mean_nll(logits, y_temp.float())
                            loss_value += args.lambda1before*model.get_entropy_loss(x_temp)
                            
                        weight_norm = torch.tensor(0.).cuda()
                        for w in model.parameters():
                              weight_norm += w.norm().pow(2)

                        loss_value+=args.l2_regularizer_weight*weight_norm    
                        loss_value/=gamma                
             
                        optimizer.zero_grad()
                        loss_value.backward()
                        optimizer.step()                        
                    else:
                        loss_value = 0.
                        gamma = 1.
                        for x_temp,y_temp in zip(xs,ys):
                            logits = model(x_temp)    
                            logits = F.log_softmax(logits, dim=1)
                            loss_value += args.lambda0*mean_nll(logits, y_temp.float())
                            loss_value += args.lambda1*model.get_gated_entropy_loss(x_temp)
                            
                        weight_norm = torch.tensor(0.).cuda()
                        for w in model.parameters():
                              weight_norm += w.norm().pow(2)

                        loss_value+=args.l2_regularizer_weight*weight_norm    
                        loss_value/=gamma                
                        
                        if args.use_gate_net_flag:
                            optimizer_gate.zero_grad()
                            loss_value.backward()

                            optimizer_gate.step()
                        else:
                            optimizer_classifier.zero_grad()
                            loss_value.backward()
                            optimizer_classifier.step()

                    if epoch%args.mask_update_every==args.mask_update_every-1:
                        model.update_mask()
                        print(f"mask={(model.get_mask()>0.5).long().sum()}",end=", ") 
                        
                    y_ = model(x_in)
                    if args.regression:
                        rmse_train,pcor_train = metric_regression(y_,y_in)
                    else:                                                    
                        acc_train = np.float(eval_accuracy(y_,y_in))     
                    if writer: writer.add_scalar("Mask/train" ,(model.get_mask()>0.5).long().sum(), global_iter)                    
                    
                #PRJ-IRMv1
                if args._type_solver.upper()=="PIRM":
                    if epoch%args.mask_update_every==args.mask_update_every-1:
                        model.update_mask()
                        print(f"mask={(model.get_mask()>0.5).long().sum()}",end=", ")                     
                    
                    
                    gamma     = 1.0
                    if(epoch>=args.steps_threshold):
                        gamma = args.gamma_new   
                        
                    loss_value = 0.
                    for x_temp,y_temp in zip(xs,ys):
                        logits = model(x_temp)    
                        logits = F.log_softmax(logits, dim=1)
                        loss_value += mean_nll(logits, y_temp.float())
                        loss_value += gamma*penalty(logits, y_temp.float())
                    weight_norm = torch.tensor(0.).cuda()
                    for w in model.parameters():
                          weight_norm += w.norm().pow(2)

                    loss_value+=args.l2_regularizer_weight*weight_norm    
                    loss_value/=gamma                
                    
                    if epoch<args.net_freeze_epoch:
                        loss_value = args.lambda0before*loss_value + args.lambda1before*model.get_entropy_loss(x_envs)               
                        optimizer.zero_grad()
                        loss_value.backward()
                        optimizer.step()                        
                    else:
                        loss_value = args.lambda0*loss_value + args.lambda1*model.get_gated_entropy_loss(x_envs)
                        if args.use_gate_net_flag:
                            optimizer_gate.zero_grad()
                            loss_value.backward()
                            optimizer_gate.step()
                        else:
                            optimizer_classifier.zero_grad()
                            loss_value.backward()
                            optimizer_classifier.step()
                        
                    y_ = model(x_in)
                    if args.regression:
                        rmse_train,pcor_train = metric_regression(y_,y_in)
                    else:                                                    
                        acc_train = np.float(eval_accuracy(y_,y_in))
                    if writer: writer.add_scalar("Mask/train" ,(model.get_mask()>0.5).long().sum(), global_iter)
                    
                #IRMv1
                if args._type_solver.upper()=="IRM":
                    gamma     = 1.0
                    if(epoch>=args.steps_threshold):
                        gamma = args.gamma_new  
                        
                    loss_value = 0.
                    for x_temp,y_temp in zip(xs,ys):
                        logits = model(x_temp)    
                        logits = F.log_softmax(logits, dim=1)
                        loss_value += mean_nll(logits, y_temp.float())
                        loss_value += gamma*penalty(logits, y_temp.float())
                        
                    weight_norm = torch.tensor(0.).cuda()
                    for w in model.parameters():
                          weight_norm += w.norm().pow(2)
                            
                    loss_value+=args.l2_regularizer_weight*weight_norm    
                    loss_value/=gamma
                    
                    if args._type in ['gated']:
                        loss_value += args.lambda1*model.get_entropy_loss(x_envs)                    

                    optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()

                    y_ = model(x_in)
                    if args.regression:
                        rmse_train,pcor_train = metric_regression(y_,y_in)
                    else:                                                    
                        acc_train = np.float(eval_accuracy(y_,y_in))
                
                
                #IRMG output computation
                if args._type_solver.upper()=="IRMG":                
                    y_ = combine_models(model_list, x)
                    loss_value = model_list[which_env].eval_loss(y_,y)


                    optimizer_list[which_env].zero_grad()
                    loss_value.backward()
                    optimizer_list[which_env].step()

                    # computing training accuracy
                    y_ = combine_models(model_list, x_in)
                    if args.regression:
                        rmse_train,pcor_train = metric_regression(y_,y_in)
                    else:                                                    
                        acc_train = np.float(eval_accuracy(y_,y_in))
                    model = lambda _x: combine_models(model_list, _x)
                    
                    
                #Loggging
                y_te_ = model(x_in_te)
                # if args.regression:
                #     rmse_train_te,pcor_train_te = metric_regression(y_te_,y_in_te)
                # else:                                
                #     acc_train_te = np.float(eval_accuracy(y_te_,y_in_te).detach())                    

                #Experiment Logging
                try:
                    _counter+=1
                except:
                    _counter=0                    

                    
                # if args._type_solver.upper() in ['PIRM',"PERM"]: 
                if args._type in ['gated']: 
                    ngates = torch.sum(model.get_mask()>.5).detach().item()
                else:
                    ngates = 0
                                       
                if args.regression:
                    rmse_train_tr,pcor_train_tr = metric_regression(y_,y_in)
                    # rmse_train_env, pcor_train_env = metric_regression(y_env,y)
                    rmse_train_te,pcor_train_te = metric_regression(y_te_,y_in_te)
                    # metrics = {
                    #     f'Loss-{kenv}/train':loss_value
                    #     ,f"rmse-{kenv}/train_tr" :rmse_train_tr
                    #     ,f"pcor-{kenv}/train_tr" :pcor_train_tr
                    #     # ,f"rmse-{kenv}/train_env" :rmse_train_env
                    #     # ,f"pcor-{kenv}/train_env" :pcor_train_env
                    #     ,f"rmse-{kenv}/train_te" :rmse_train_te
                    #     ,f"pcor-{kenv}/train_te" :pcor_train_te
                    #     ,f"ngates-{kenv}/train": ngates
                    #     ,f"kenv": kenv
                    #     ,f"env_iter": env_iter
                    # }
                    score = max(rmse_train_tr,rmse_train_te)
                    metrics = {
                        f'Loss_train':loss_value
                        ,f"rmse_train_tr" :rmse_train_tr
                        ,f"pcor_train_tr" :pcor_train_tr
                        # ,f"rmse-{kenv}/train_env" :rmse_train_env
                        # ,f"pcor-{kenv}/train_env" :pcor_train_env
                        ,f"rmse_train_te" :rmse_train_te
                        ,f"pcor_train_te" :pcor_train_te
                        ,f"ngates_train": ngates
                        ,f"kenv": which_env
                        ,f"env_iter": global_iter
                        ,"epoch": epoch
                        , 'rnd': args.rnd
                        , 'score':score
                    }
                    
                else:                        
                    acc_train_tr = np.float(eval_accuracy(y_,y_in))
                    # acc_train_env = np.float(eval_accuracy(y_env,y).detach())
                    acc_train_te = np.float(eval_accuracy(y_te_,y_in_te).detach())
                    # metrics = {
                    #     f'Loss-{kenv}/train':loss_value
                    #     ,f"Accuracy-{kenv}/train_tr" :acc_train_tr
                    #     # ,f"Accuracy-{kenv}/train_env" :acc_train_env
                    #     ,f"Accuracy-{kenv}/train_te" :acc_train_te
                    #     ,f"ngates-{kenv}/train": ngates
                    #     ,f"kenv": kenv
                    #     ,f"env_iter": env_iter                            
                    # }     
                    score = min(acc_train_tr,acc_train_te)
                    metrics = {
                        f'Loss_train':loss_value
                        ,f"Accuracy_train_tr" :acc_train_tr
                        # ,f"Accuracy-{kenv}/train_env" :acc_train_env
                        ,f"Accuracy_train_te" :acc_train_te
                        ,f"ngates_train": ngates
                        ,f"kenv": which_env
                        ,f"env_iter": global_iter
                        ,"epoch": epoch
                        , 'rnd': args.rnd
                        , 'score':score
                    }                  
                    
                                     
                # args.logger.log_metrics(metrics, step=env_iter)
                args.logger.log_metrics(metrics, step=_counter)
                if args.verbose:
                    print(f"metrics={metrics}")
                args.logger.save()
                
                args.dblog.insert(metrics, 'gib_runs')
                # try: args.dblog.insert(metrics, 'gib_runs')
                # except: pass
                    
                                           
                # register performances
                if writer: writer.add_scalar("Loss/train", loss_value, global_iter)
                if args.regression:
                    if writer: writer.add_scalar("rmse/train" ,rmse_train, global_iter)
                    if writer: writer.add_scalar("rmse/test_while_training" ,rmse_train_te, global_iter)
                    if writer: writer.add_scalar("pcor/train" ,pcor_train, global_iter)
                    if writer: writer.add_scalar("pcor/test_while_training" ,pcor_train_te, global_iter)
                    if writer: writer.add_scalar("ngates_train" ,ngates, global_iter)
                else:
                    if writer: writer.add_scalar("Accuracy/train" ,acc_train, global_iter)
                    if writer: writer.add_scalar("Accuracy/test_while_training" ,acc_train_te, global_iter)
                    if writer: writer.add_scalar("ngates_train" ,ngates, global_iter)

                    
        return 'finished'
# --------------------------------------------------------    
from sklearn.utils import shuffle

def train_as_one(data_tuple_list_forget,data_tuple_test_forget, model_list, args, writer=None):
        n_e  = len(data_tuple_list_forget) 
        x_in,y_in,e_in = concat_envs(data_tuple_list_forget) 
        
        y_in = torch.Tensor(y_in)
        x_in = torch.Tensor(x_in)
        e_in = torch.Tensor(e_in)
        
        x_in_te = data_tuple_test_forget[0]
        y_in_te = data_tuple_test_forget[1]
        y_in_te = torch.Tensor(y_in_te)
        x_in_te = torch.Tensor(x_in_te)                
        
        set_models(model_list,train_flag=True)

        # initialize optimizer for all the environments and representation learner and store it in a list
        if args._type_solver.upper() in ['IRMG']:
            optimizer_list = get_optimizer_factory(args)(model_list,args)
            model = lambda _x: combine_models(model_list, _x)
        else:    
            if type(model_list)==list:
                model = model_list[0]
            else:
                model = model_list
            optimizer = get_optimizer(model,args)
            # if args._type_solver.upper() in ['PIRM',"PERM","PIRM"]:                
            if args._type in ['gated']:                
                optimizer_gate = get_optimizer_gate(model,args)
                optimizer_classifier = get_optimizer_classifier(model,args)            
            scheduler = None
            if args.scheduler_flag :
                scheduler = get_scheduler(optimizer,args)


        ####### train
        num_examples = x_in.shape[0]
        global_iter           = 0
        w = torch.tensor(1.).cuda().requires_grad_() 
        first_time = True
        for epoch in range(args.num_epochs):    
            if epoch and not args.check_loop_fn is None:
                if args.check_loop_fn(args,score,epoch):
                    return 'stopped'
            if args.verbose: print ("Epoch: {}/{}({})".format(epoch,args.num_epochs,n_e))
            else: print(".",end="")
            x_in,y_in,e_in = shuffle(x_in,y_in,e_in)
            if args.parallel_every_epoch_flag:
                global_iter +=1
            for count, offset in enumerate(range(0,num_examples, args.batch_size)): 
                end = offset + args.batch_size
                if not args.parallel_every_epoch_flag:
                    global_iter +=1

                assert(args._type_solver.upper() in ["ERM"])
                
                #ERM
                if args._type_solver.upper()=="ERM":         
                    y_pred = model(x_in)
                    loss_value = model.eval_loss(y_pred,y_in)

                    # if args._type in ['gated']:
                    #     loss_value += args.lambda0*model.gate3.get_loss()
                    #     loss_value += args.lambda1*model.get_entropy_loss(x_in)
                    # optimizer.zero_grad()
                    # loss_value.backward()
                    # optimizer.step()

                    if epoch<args.net_freeze_epoch and args._type in ['gated']:
                        loss_value = args.lambda0before*loss_value + args.lambda1before*model.get_entropy_loss(x_in)               
                        optimizer.zero_grad()
                        loss_value.backward()
                        optimizer.step()                        
                    else:
                        loss_value = args.lambda0*loss_value + args.lambda1*model.get_gated_entropy_loss(x_in)
                        if args.use_gate_net_flag:
                            optimizer_gate.zero_grad()
                            loss_value.backward()
                            optimizer_gate.step()
                        else:
                            optimizer_classifier.zero_grad()
                            loss_value.backward()
                            optimizer_classifier.step()                    
                    
                    y_ = model(x_in)
                    if args.regression:
                        rmse_train,pcor_train = metric_regression(y_,y_in)
                    else:                                
                        acc_train = np.float(eval_accuracy(y_,y_in))
                        
                    if args._type in ['gated'] and epoch%args.mask_update_every==args.mask_update_every-1:
                        model.update_mask()
                        print(f"mask={(model.get_mask()>0.5).long().sum()}",end=", ")                         
                    
                #Loggging
                y_te_ = model(x_in_te)

                #Experiment Logging
                try:
                    _counter+=1
                except:
                    _counter=0                    

                    
                # if args._type_solver.upper() in ['PIRM',"PERM"]: 
                if args._type in ['gated']: 
                    ngates = torch.sum(model.get_mask()>.5).detach().item()
                else:
                    ngates = 0
                                       
                if args.regression:
                    rmse_train_tr,pcor_train_tr = metric_regression(y_,y_in)
                    rmse_train_te,pcor_train_te = metric_regression(y_te_,y_in_te)
                    score = max(rmse_train_tr,rmse_train_te)
                    metrics = {
                        f'Loss_train':loss_value
                        ,f"rmse_train_tr" :rmse_train_tr
                        ,f"pcor_train_tr" :pcor_train_tr
                        ,f"rmse_train_te" :rmse_train_te
                        ,f"pcor_train_te" :pcor_train_te
                        ,f"ngates_train": ngates
                        # ,f"kenv": which_env
                        ,f"env_iter": global_iter
                        ,"epoch": epoch
                        , 'rnd': args.rnd
                        , 'score':score
                    }
                    
                else:                        
                    acc_train_tr = np.float(eval_accuracy(y_,y_in))
                    acc_train_te = np.float(eval_accuracy(y_te_,y_in_te).detach())
                    score = min(acc_train_tr,acc_train_te)
                    metrics = {
                        f'Loss_train':loss_value
                        ,f"Accuracy_train_tr" :acc_train_tr
                        ,f"Accuracy_train_te" :acc_train_te
                        ,f"ngates_train": ngates
                        # ,f"kenv": which_env
                        ,f"env_iter": global_iter
                        ,"epoch": epoch
                        , 'rnd': args.rnd
                        , 'score':score
                    }                  
                    
                                     
                args.logger.log_metrics(metrics, step=_counter)
                if args.verbose:
                    print(f"metrics={metrics}")
                args.logger.save()
                
                args.dblog.insert(metrics, 'gib_runs')                    
                                           
                # register performances
                if writer: writer.add_scalar("Loss/train", loss_value, global_iter)
                if args.regression:
                    if writer: writer.add_scalar("rmse/train" ,rmse_train, global_iter)
                    if writer: writer.add_scalar("rmse/test_while_training" ,rmse_train_te, global_iter)
                    if writer: writer.add_scalar("pcor/train" ,pcor_train, global_iter)
                    if writer: writer.add_scalar("pcor/test_while_training" ,pcor_train_te, global_iter)
                    if writer: writer.add_scalar("ngates_train" ,ngates, global_iter)
                else:
                    if writer: writer.add_scalar("Accuracy/train" ,acc_train, global_iter)
                    if writer: writer.add_scalar("Accuracy/test_while_training" ,acc_train_te, global_iter)
                    if writer: writer.add_scalar("ngates_train" ,ngates, global_iter)

                    
        return 'finished'    
    
    
    
    
# ------------------------------------------------------------    
def train_factory(args):
    # if args.sequential_flag:
    #     return train_sequential
    # else:
    #     return train_together
    if args.training_type in ['sequential']:
        return train_sequential
    if args.training_type in ['parallel']:
        return train_together
    if args.training_type in ['one']:
        return train_as_one
def evaluate_factory(args):
    if args._type_solver.upper()=="IRMG":
        return evaluate_list
    else:
        return evaluate_single

def evaluate_list(data_tuple_test, data_tuple_list, model_list):
    x_test = data_tuple_test[0]
    y_test = data_tuple_test[1]
    y_test = torch.Tensor(y_test)
    x_test = torch.Tensor(x_test)

    x_in,y_in,e_in = concat_envs(data_tuple_list)
    y_in = torch.Tensor(y_in)
    x_in = torch.Tensor(x_in)

    set_models(model_list,train_flag=False)
    
    test_acc, train_acc = 0,0
    train_rmse, train_pcor = 0,0
    test_rmse, test_pcor = 0,0

    ytr_ = combine_models(model_list, x_in)
    if model_list[0].regression:
        train_rmse, train_pcor = metric_regression(ytr_,y_in)
        train_rmse, train_pcor = np.float(train_rmse), np.float(train_pcor)
    else:
        train_acc = np.float(eval_accuracy(ytr_,y_in))

    yts_ = combine_models(model_list, x_test)
    if  model_list[0].regression:
        test_rmse, test_pcor = metric_regression(yts_,y_test)
        test_rmse, test_pcor = np.float(test_rmse), np.float(test_pcor)
    else:
        test_acc = np.float(eval_accuracy(yts_,y_test))

    if model_list[0].regression:
        return (test_rmse, test_pcor), (train_rmse, train_pcor)
    else:
        return test_acc, train_acc


def evaluate_single(data_tuple_test, data_tuple_list, model):
    x_test = data_tuple_test[0]
    y_test = data_tuple_test[1]
    y_test = torch.Tensor(y_test)
    x_test = torch.Tensor(x_test)

    x_in,y_in,e_in = concat_envs(data_tuple_list)
    y_in = torch.Tensor(y_in)
    x_in = torch.Tensor(x_in)

    set_models(model,train_flag=False)
     
    test_acc, train_acc = 0,0
    train_rmse, train_pcor = 0,0
    test_rmse, test_pcor = 0,0

    ytr_ = model(x_in)
    if model.regression:
        train_rmse, train_pcor = metric_regression(ytr_,y_in)
        train_rmse, train_pcor = np.float(train_rmse), np.float(train_pcor)
    else:
        train_acc = np.float(eval_accuracy(ytr_,y_in))

    yts_ = model(x_test)
    if model.regression:
        test_rmse, test_pcor = metric_regression(yts_,y_test)
        test_rmse, test_pcor = np.float(test_rmse), np.float(test_pcor)
    else:
        test_acc = np.float(eval_accuracy(yts_,y_test))

    if model.regression:
        return (test_rmse, test_pcor), (train_rmse, train_pcor)
    else:
        return test_acc, train_acc

        
