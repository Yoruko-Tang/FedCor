#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import copy

from utils import average_weights

import numpy as np
from tqdm import tqdm




def ce_criterion(pred, target, *args):
    ce_loss = F.cross_entropy(pred, target)
    return ce_loss, float(ce_loss)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, global_round = 0,verbose = None):
        self.args = args
        # self.logger = logger
        self.trainloader, self.testloader = self.train_test(
            dataset, list(idxs))
        self.device = 'cuda:'+args.gpu if args.gpu else 'cpu'
        self.criterion = ce_criterion
        self.test_criterion = ce_criterion
        self.global_round = global_round
        self.mu = args.mu
        self.verbose = args.verbose if verbose is None else verbose
        

    def train_test(self, dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset and user indexes.
        """
        # split indexes for train, and test (80, 20)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]
        #print(idxs_test)
        # c = Counter(np.array(dataset.targets)[idxs_test])
        # print(c)
        # input()
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max([int(len(idxs_test)/10),10]), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model):
        # Update model's weights or gates
        # Set mode to train model
        self.net = model.to(self.device)
        self.init_net = copy.deepcopy(self.net)
        init_accuracy,init_test_loss = self.inference(self.net)
        self.net.train()
        np = self.net.parameters()
        # Set optimizer for the local updates

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(np, lr=self.args.lr,
                                        momentum=self.args.momentum,weight_decay=self.args.reg)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(np, lr=self.args.lr,
                                        weight_decay=self.args.reg)
        

        for iter in range(self.args.local_ep):
            for batch_idx, (datas, labels) in enumerate(self.trainloader):
                
                datas, labels = datas.to(self.device), labels.to(self.device)

                self.net.zero_grad()
                output = self.net(datas)
                total_loss,celoss = self.criterion(output, labels)
                if self.args.FedProx:
                    PLoss = self.Proxy_Loss()
                    total_loss += 0.5*self.mu*PLoss
                total_loss.backward()
                optimizer.step()

            if self.verbose:
                print('| Global Round : {} | Local Epoch : {} |\tLoss: {:.4f}\tCE_Loss: {:.4f}\tProxy_Loss: {:.4f}'.format(
                        self.global_round, iter, total_loss.item(),celoss,PLoss if self.args.FedProx else 0.0))
                # self.logger.add_scalar('weight_loss', total_loss.item())
        
        
        test_accuracy,test_loss = self.inference(self.net)

        return self.net.state_dict(), test_loss,init_test_loss

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (datas, labels) in enumerate(self.testloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            # print(labels)
            # input()
            # Inference
            outputs = model(datas)
            batch_loss,_ = self.test_criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)

    def Proxy_Loss(self):
        loss = 0.
        init_state_dict = self.init_net.state_dict()
        for name,p in self.net.named_parameters():
            if 'weight' in name or 'bias' in name:
                loss += torch.sum((p-init_state_dict[name])**2)
        return loss


def train_federated_learning(args,epoch,global_model,idxs_users,train_dataset,user_groups,verbose = False):
    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    local_weights = []
    for _ in range(args.num_users):
        local_weights.append(copy.deepcopy(global_model.state_dict()))
    local_weights = np.array(local_weights)
    global_model.train()
    for idx in idxs_users:
        local_model = copy.deepcopy(global_model)
        local_update = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx] ,global_round = epoch,verbose=verbose)
        
        w,_,_ = local_update.update_weights(model=local_model)
        local_weights[idx]=copy.deepcopy(w)

    # update global weights
    if args.global_average:
        global_weights = average_weights(local_weights)
    else:
        global_weights = average_weights(local_weights[idxs_users])

    global_model.load_state_dict(global_weights)

    # Calculate test accuracy over all users at every epoch
    list_acc, list_loss = [], []
    global_model.eval()
    for idx in range(args.num_users):
        local_model = copy.deepcopy(global_model)
        local_update = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx],verbose=verbose)
        acc, loss = local_update.inference(model=local_model)
        list_acc.append(acc)
        list_loss.append(loss)
    return sum(list_acc)/len(list_acc),list_loss

def federated_train_all(args,global_model,train_dataset,user_groups):
    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    global_model.train()
    local_weights = []
    for idx in tqdm(range(args.num_users)):
        local_model = copy.deepcopy(global_model)
        local_update = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx] ,verbose=False)
        
        w,_,_ = local_update.update_weights(model=local_model)
        local_weights.append(copy.deepcopy(w))
    return np.array(local_weights)

def federated_test_idx(args,global_model,idxs_users,train_dataset,user_groups):
    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    global_model.eval()
    list_acc, list_loss = [], []
    local_model = copy.deepcopy(global_model)
    for idx in idxs_users:
        local_update = LocalUpdate(args=args, dataset=train_dataset,
                                idxs=user_groups[idx])
        acc, loss = local_update.inference(model=local_model)
        list_acc.append(acc)
        list_loss.append(loss)
    return list_acc,list_loss

def federated_train_worker(args,global_model,idxs,train_dataset,user_groups,local_states,local_weights,epoch_global_losses,epoch_local_losses):
    if isinstance(idxs,int):
        idxs = [idxs,]
    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    for idx in idxs:
        local_model = copy.deepcopy(global_model)
        local_update = LocalUpdate(args=args, dataset=train_dataset,
                                idxs=user_groups[idx])
        w,test_loss,init_test_loss = local_update.update_weights(model=local_model)
        
        local_states[idx] = copy.deepcopy(local_model.Get_Local_State_Dict())
        local_weights[idx]=copy.deepcopy(w)
        epoch_global_losses.append(init_test_loss)# TAKE CARE: this is the test loss evaluated on the (t-1)-th global weights!
        epoch_local_losses.append(test_loss)
    

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    model.to(device)
    criterion = F.cross_entropy
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (datas, labels) in enumerate(testloader):
        datas, labels = datas.to(device), labels.to(device)

        # Inference
        outputs = model(datas)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss/(batch_idx+1)
