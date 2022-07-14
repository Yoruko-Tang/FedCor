#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import torch.multiprocessing

# from language_utils import get_word_emb_arr

from options import args_parser
from update import LocalUpdate,test_inference,train_federated_learning,federated_test_idx
from models import MLP, NaiveCNN, BNCNN, ResNet,RNN
from utils import get_dataset, average_weights, exp_details,setup_seed
from mvnt import MVN_Test
import GPR
from GPR import Kernel_GPR,TrainGPR


import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from math import ceil



if __name__ == '__main__':
    os.environ["OUTDATED_IGNORE"]='1'
    torch.multiprocessing.set_start_method('spawn')
    start_time = time.time()
    # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('./logs')

    args = args_parser()
    gargs = copy.deepcopy(args)
    exp_details(args)
    if not args.iid:
        base_file = './save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_{}[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    'sp' if args.alpha is None else 'alpha',args.shards_per_client if args.alpha is None else args.alpha,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    else:
        base_file = './save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    if not os.path.exists(base_file):
        os.makedirs(base_file)
    
    if args.afl:
        file_name = base_file+'/afl'
    elif args.power_d:
        file_name = base_file+'/powerd_d[{}]'.format(args.d)
    elif not args.gpr_selection:
        file_name = base_file+'/random'
    else:
        file_name = base_file+'/gpr[int{}_gp{}_norm{}]_{}[{}]'.\
            format(args.GPR_interval,args.group_size,args.poly_norm,
            args.discount_method,args.loss_power if args.discount_method=='loss' else args.discount)
    


    

    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    if args.gpu:
        torch.cuda.set_device(device)
    if gargs.seed is None or gargs.iid:
        gargs.seed = [None,]
    for seed in gargs.seed:
        args = copy.deepcopy(gargs)# recover the args
        print("Start with Random Seed: {}".format(seed))
        # load dataset and user groups
        train_dataset, test_dataset, user_groups, user_groups_test,weights = get_dataset(args,seed)
        # weights /=np.sum(weights)
        if seed is not None:
            setup_seed(seed)
        data_size = train_dataset[0][0].shape
        # BUILD MODEL
        if args.model == 'cnn':
            # Naive Convolutional neural netork
            global_model = NaiveCNN(args=args,input_shape = data_size,final_pool=False)
        
        elif args.model == 'bncnn':
            # Convolutional neural network with batch normalization
            global_model = BNCNN(args = args, input_shape = data_size)

        elif args.model == 'mlp' or args.model == 'log':
            # Multi-layer preceptron
            len_in = 1
            for x in data_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=args.mlp_layers if args.model=='mlp' else [],
                                dim_out=args.num_classes)
        elif args.model == 'resnet':
            global_model = ResNet(args.depth,args.num_classes)
        elif args.model == 'rnn':
            if args.dataset=='shake':
                global_model = RNN(256,args.num_classes)
            else:
                # emb_arr,_,_= get_word_emb_arr('./data/sent140/embs.json')
                global_model = RNN(256,args.num_classes,300,True,128)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)

        # Build GP
        if args.gpr:
            if args.kernel=='Poly':
                gpr = Kernel_GPR(args.num_users,dimension = args.dimension,init_noise=0.01,
                                    order = 1,Normalize = args.poly_norm,kernel=GPR.Poly_Kernel,loss_type= args.train_method)
            elif args.kernel=='SE':
                gpr = Kernel_GPR(args.num_users,dimension = args.dimension,init_noise=0.01,kernel=GPR.SE_Kernel,loss_type= args.train_method)
            else:
                gpr = GPR.Matrix_GPR(args.num_users,loss_type=args.train_method)
            # gpr.to(device)

        # copy weights
        global_weights = global_model.state_dict()
        local_weights = []# store local weights of all users for averaging
        local_states = []# store local states of all users, these parameters should not be uploaded

        
        for i in range(args.num_users):
            local_states.append(copy.deepcopy(global_model.Get_Local_State_Dict()))
            local_weights.append(copy.deepcopy(global_weights))

        local_states = np.array(local_states)
        local_weights = np.array(local_weights)

        # Training
        train_loss, train_accuracy = [], []
        test_loss,test_accuracy = [],[]
        max_accuracy=0.0

        local_losses = []# test losses evaluated on local models(before averaging)
        # global_losses = []# test losses evaluated on global models(after averaging)
        chosen_clients = []# chosen clients on each epoch
        gt_global_losses = []# test losses on global models(after averaging) over all clients
        gpr_data = []# GPR Training data
        print_every = 1
        init_mu = args.mu

        
        gpr_idxs_users = None
        gpr_loss_decrease = []
        gpr_acc_improve = []
        rand_loss_decrease = []
        rand_acc_improve = []

        predict_losses = []
        offpolicy_losses = []
        # mu = []
        sigma = []
        sigma_gt=[]


        # Test the global model before training
        list_acc, list_loss = federated_test_idx(args,global_model,
                                                list(range(args.num_users)),
                                                train_dataset,user_groups)
        # global_model.eval()
        # local_model = copy.deepcopy(global_model).to(device)
        # for idx in range(args.num_users):
        #     local_update = LocalUpdate(args=args, dataset=train_dataset,
        #                                 idxs=user_groups[idx])
        #     acc, loss = local_update.inference(model=local_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        gt_global_losses.append(list_loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        
        if args.afl:
            AFL_Valuation = np.array(list_loss)*np.sqrt(weights*len(train_dataset))

        # gpr_loss_data = None
        for epoch in tqdm(range(args.epochs)):
            print('\n | Global Training Round : {} |\n'.format(epoch+1))
            epoch_global_losses = []
            epoch_local_losses = []
            global_model.train()
            if args.dataset=='cifar' or epoch in args.schedule:
                args.lr*=args.lr_decay
                
            
                

            if gpr_idxs_users is not None and not args.gpr_selection:
                # Testing off-policy selection
                if args.verbose:
                    print("Training with GPR Selection:")
                gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                                    copy.deepcopy(global_model),gpr_idxs_users,train_dataset,user_groups)
                gpr_loss_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                                np.expand_dims(np.array(gpr_loss)-np.array(gt_global_losses[-1]),1),
                                                np.ones([args.num_users,1])],1)
                predict_loss,_,_=gpr.Predict_Loss(gpr_loss_data,gpr_idxs_users,np.delete(list(range(args.num_users)),gpr_idxs_users))
                print("GPR Predict Off-Policy Loss:{:.4f}".format(predict_loss))
                offpolicy_losses.append(predict_loss)

                gpr_dloss = np.sum((np.array(gpr_loss)-np.array(gt_global_losses[-1]))*weights)
                gpr_loss_decrease.append(gpr_dloss)
                gpr_acc_improve.append(gpr_acc-train_accuracy[-1])
                if args.verbose:
                    print("Training with {} Selection".format('Random' if not args.power_d else 'Power-D'))
            

            
            m = max(int(args.frac * args.num_users), 1)

            if args.afl:
                delete_num = int(args.alpha1*args.num_users)
                sel_num = int((1-args.alpha3)*m)
                tmp_value = np.vstack([np.arange(args.num_users),AFL_Valuation])
                tmp_value = tmp_value[:,tmp_value[1,:].argsort()]
                prob = np.exp(args.alpha2*tmp_value[1,delete_num:])
                prob = prob/np.sum(prob)
                sel1 = np.random.choice(np.array(tmp_value[0,delete_num:],dtype=np.int64),sel_num,replace=False,p=prob)
                remain = set(np.arange(args.num_users))-set(sel1)
                sel2 = np.random.choice(list(remain),m-sel_num,replace = False)
                idxs_users = np.append(sel1,sel2)


            elif args.power_d:
                # use power_d algorithm
                A = np.random.choice(range(args.num_users), args.d, replace=False,p=weights)
                idxs_users = A[np.argsort(np.array(gt_global_losses[-1])[A])[-m:]]
            elif not args.gpr_selection or gpr_idxs_users is None:
                # random selection
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            else:
                # FedGP
                idxs_users = copy.deepcopy(gpr_idxs_users)

            chosen_clients.append(idxs_users)
            
            for idx in idxs_users:
                local_model = copy.deepcopy(global_model)
                local_update = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx] ,global_round = epoch)
                w,test_loss,init_test_loss = local_update.update_weights(model=local_model)
                
                local_states[idx] = copy.deepcopy(local_model.Get_Local_State_Dict())
                local_weights[idx]=copy.deepcopy(w)
                epoch_global_losses.append(init_test_loss)# TAKE CARE: this is the test loss evaluated on the (t-1)-th global weights!
                epoch_local_losses.append(test_loss)


            # update global weights
            if args.global_average:
                global_weights = average_weights(local_weights,omega=None)
            else:
                global_weights = average_weights(local_weights[idxs_users],omega=None)

            for i in range(args.num_users):
                local_weights[i] = copy.deepcopy(global_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            if args.afl:
                AFL_Valuation[idxs_users] = np.array(epoch_global_losses)*np.sqrt(weights[idxs_users]*len(train_dataset))
            # global_losses.append(epoch_global_losses)
            local_losses.append(epoch_local_losses)

            # dynamic mu for FedProx
            loss_avg = sum(epoch_local_losses) / len(epoch_local_losses)
            if args.dynamic_mu and epoch>0:
                if loss_avg>loss_prev:
                    args.mu+=init_mu*0.1
                else:
                    args.mu=max([args.mu-init_mu*0.1,0.0])
            loss_prev = loss_avg
            train_loss.append(loss_avg)

            # calculate test accuracy over all users
            list_acc, list_loss = federated_test_idx(args,global_model,
                                                    list(range(args.num_users)),
                                                    train_dataset,user_groups)
            # global_model.eval()
            # local_model = copy.deepcopy(global_model).to(device)
            # for idx in range(args.num_users):
            #     local_update = LocalUpdate(args=args, dataset=train_dataset,
            #                             idxs=user_groups[idx])
            #     acc, loss = local_update.inference(model=local_model)
            #     list_acc.append(acc)
            #     list_loss.append(loss)
            gt_global_losses.append(list_loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # calculate the advantage in off-policy
            if gpr_idxs_users is not None and not args.gpr_selection:
                rand_loss_decrease.append(np.sum((np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]))*weights))
                rand_acc_improve.append(train_accuracy[-1]-train_accuracy[-2])
                print("Advantage:",gpr_loss_decrease[-1]-rand_loss_decrease[-1])
            
            # test prediction accuracy of GP model
            if args.gpr and epoch>args.warmup:
                test_idx = np.random.choice(range(args.num_users), m, replace=False)
                test_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                            np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                            np.ones([args.num_users,1])],1)
                pred_idx = np.delete(list(range(args.num_users)),test_idx)
                
                predict_loss,mu_p,sigma_p = gpr.Predict_Loss(test_data,test_idx,pred_idx)
                print("GPR Predict relative Loss:{:.4f}".format(predict_loss))
                predict_losses.append(predict_loss)
                # mu.append(mu_p.detach().numpy())
                # sigma.append(sigma_p.detach().numpy())
                

            

            # train and exploit GPR
            if args.gpr:
                if epoch<=args.warmup and epoch>=args.gpr_begin:# warm-up
                    gpr.update_loss(np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                                    np.expand_dims(np.array(gt_global_losses[-1]),1)],1))
                    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                                    np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                                    np.ones([args.num_users,1])],1)
                    gpr_data.append(epoch_gpr_data)
                    print("Training GPR")
                    TrainGPR(gpr,gpr_data[max([(epoch-args.gpr_begin-args.group_size+1),0]):epoch-args.gpr_begin+1],
                            args.train_method,lr = 1e-2,llr = 0.0,gamma = args.GPR_gamma,max_epoches=args.GPR_Epoch+50,schedule_lr=False,verbose=args.verbose)

                elif epoch>args.warmup and epoch%args.GPR_interval==0:# normal and optimization round
                    gpr.update_loss(np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                                    np.expand_dims(np.array(gt_global_losses[-1]),1)],1))
                    gpr.Reset_Discount()
                    print("Training with Random Selection For GPR Training:")
                    random_idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                    gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                                        copy.deepcopy(global_model),random_idxs_users,train_dataset,user_groups)
                    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                                    np.expand_dims(np.array(gpr_loss)-np.array(gt_global_losses[-1]),1),
                                                    np.ones([args.num_users,1])],1)
                    gpr_data.append(epoch_gpr_data)
                    print("Training GPR")
                    TrainGPR(gpr,gpr_data[-ceil(args.group_size/args.GPR_interval):],
                            args.train_method,lr = 1e-2,llr = 0.0,gamma = args.GPR_gamma**args.GPR_interval,max_epoches=args.GPR_Epoch,schedule_lr=False,verbose=args.verbose)
                
                else:# normal and not optimization round
                    gpr.update_loss(np.concatenate([np.expand_dims(idxs_users,1),
                                                np.expand_dims(epoch_global_losses,1)],1))
                    gpr.update_discount(idxs_users,args.discount)
                    
                if epoch>=args.warmup:
                    gpr_idxs_users = gpr.Select_Clients(m,args.loss_power,args.epsilon_greedy,args.discount_method,weights,args.dynamic_C,args.dynamic_TH)
                    print("GPR Chosen Clients:",gpr_idxs_users)
                
                if args.mvnt and (epoch==args.warmup or (epoch%args.mvnt_interval==0 and epoch>args.warmup)):
                    mvn_file = file_name+'/MVN/{}'.format(seed)
                    if not os.path.exists(mvn_file):
                        os.makedirs(mvn_file)
                    mvn_samples=MVN_Test(args,copy.deepcopy(global_model),
                                                train_dataset,user_groups,
                                                file_name+'/MVN/{}/{}.csv'.format(seed,epoch))
                    sigma_gt.append(np.cov(mvn_samples,rowvar=False,bias = True))
                    sigma.append(gpr.Covariance().clone().detach().numpy())

                    
                
                
                

            # test inference on the global test dataset
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            if args.target_accuracy is not None:
                if test_acc>=args.target_accuracy:
                    break

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
                print('Training Loss : {}'.format(np.sum(np.array(list_loss)*weights)))
                # print('Train Accuracy: {:.2f}%'.format(100*train_accuracy[-1]))
                print("Test Accuracy: {:.2f}%\n".format(100*test_acc))
        
        
        print(' \n Results after {} global rounds of training:'.format(epoch+1))
        print("|---- Final Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
        # print("|---- Max Train Accuracy: {:.2f}%".format(100*max(train_accuracy)))
        print("|---- Max Test Accuracy: {:.2f}%".format(100*max(test_accuracy)))

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        # save the training records:
        with open(file_name+'_{}.pkl'.format(seed), 'wb') as f:
            pickle.dump([train_loss, train_accuracy,chosen_clients,
                        weights,None if not args.gpr else gpr.state_dict(),
                        gt_global_losses,test_accuracy], f)
        
        if args.mvnt:
            with open(file_name+'/MVN/{}/Sigma.pkl'.format(seed), 'wb') as f:
                pickle.dump([sigma,sigma_gt],f)

        
        
        
