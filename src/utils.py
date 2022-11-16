#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import Dirichlet_noniid
from sampling import shakespeare,sent140


import numpy as np
from numpy.random import RandomState
# from random import Random
import random




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_dataset(args,seed=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    rs = RandomState(seed)
    if args.dataset == 'cifar':
        args.num_classes = 10
        data_dir = './data/cifar10'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users,rs)
            user_groups_test = cifar_iid(test_dataset,args.num_users,rs)
        else:
            # Sample Non-IID user data from Mnist
            if args.alpha is not None:
                user_groups,_ = Dirichlet_noniid(train_dataset, args.num_users,args.alpha,rs)
                user_groups_test,_ = Dirichlet_noniid(test_dataset, args.num_users,args.alpha,rs)
            elif args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
                user_groups_test = cifar_noniid(test_dataset, args.num_users,args.shards_per_client,rs)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        args.num_classes = 10

        data_dir = './data'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users,rs)
            user_groups_test = mnist_iid(test_dataset,args.num_users,rs)
        else:
            # Sample Non-IID user data from Mnist
            if args.alpha is not None:
                user_groups,_ = Dirichlet_noniid(train_dataset, args.num_users,args.alpha,rs)
                user_groups_test,_ = Dirichlet_noniid(test_dataset, args.num_users,args.alpha,rs)
            elif args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users,rs)
                user_groups_test = mnist_noniid_unequal(test_dataset, args.num_users,rs)
            else:
                user_groups = mnist_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
                user_groups_test = mnist_noniid(test_dataset,args.num_users,args.shards_per_client,rs)
    
    elif args.dataset == 'shake':
        args.num_classes = 80
        data_dir = './data/shakespeare/'
        user_groups_test={}
        train_dataset,test_dataset,user_groups=shakespeare(data_dir,args.shards_per_client,rs)
    elif args.dataset == 'sent':
        args.num_classes = 2
        data_dir = './data/sent140/'
        user_groups_test={}
        train_dataset,test_dataset,user_groups=sent140(data_dir,args.shards_per_client,rs)
        
    else:
        raise RuntimeError("Not registered dataset! Please register it in utils.py")
    
    args.num_users=len(user_groups.keys())
    weights = []
    for i in range(args.num_users):
        weights.append(len(user_groups[i])/len(train_dataset))
    
    
    return train_dataset, test_dataset, user_groups, user_groups_test,np.array(weights)


def average_weights(w,omega=None):
    """
    Returns the average of the weights.
    """
    if omega is None:
        # default : all weights are equal
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
        #omega = np.ones(len(w))
    omega = omega/np.sum(omega)
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        avg_molecule = 0
        for i in range(len(w)):
            avg_molecule+=w[i][key]*omega[i]
        w_avg[key] = copy.deepcopy(avg_molecule)
    return w_avg





def exp_details(args):
    print('\nExperimental details:')
    print('    Model     : {}'.format(args.model))
    print('    Optimizer : {}'.format(args.optimizer))
    print('    Learning  : {}'.format(args.lr))
    print('    Global Rounds   : {}'.format(args.epochs))

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print('    Fraction of users  : {}'.format(args.frac))
    print('    Local Batch size   : {}'.format(args.local_bs))
    print('    Local Epochs       : {}\n'.format(args.local_ep))
    if args.FedProx:
        print('    Algorithm    :    FedProx({})'.format(args.mu))
    else:
        print('    Algorithm    :    FedAvg')
    return

if __name__ == "__main__":
    from options import args_parser
    import matplotlib.pyplot as plt
    ALL_LETTERS = np.array(list("\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"))
    args = args_parser()
    args.dataset = 'sent'
    args.shards_per_client=1
    print(args.dataset)
    train_dataset, test_dataset, user_groups, user_groups_test,weights = get_dataset(args)
    print(len(train_dataset))
    print(len(test_dataset))
    # print(train_dataset[100][0].max())
    # print(''.join(ALL_LETTERS[train_dataset[0][0].numpy()].tolist()))
    # print(''.join(ALL_LETTERS[train_dataset[0][1].numpy()].tolist()))
    print(args.num_users)
    plt.hist(weights,bins=20)
    plt.show()
    
