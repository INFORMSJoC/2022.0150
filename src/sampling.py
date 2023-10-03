#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    print('datasize of every client: ' + str(num_items))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # rank
    # with open('../data/cifar100/order', 'r') as f:
    #     order = eval(f.read())
    # order_users = sorted(dict_users.items(), key=lambda x: sum([order.index(i) for i in x[1]]))
    # dict_users = {}
    # for item in order_users:
    #     dict_users[item[0]] = item[1]
    #     print(sum([order.index(i) for i in item[1]]))
    return dict_users


def cifar100_clorder(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    print('datasize of every client: ' + str(num_items))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    with open('../data/cifar100/order', 'r') as f:
        print('loading order...')
        file = f.read()
        order = eval(file)

    for i in range(num_users):
        dict_users[i] = order[i * num_items:(i + 1) * num_items]
        # dict_users[i] = order[i::num_users]
    return dict_users


def cifar100_partdata(dataset, num_users, alpha):
    num_items = int(len(dataset) / num_users)
    # select num_items data samples from pool
    # alpha = 1  iid distributions
    # alpha = 0  difficulty-increasing distributions
    poolsize = int(num_users ** alpha) * num_items
    print('datasize of every client: ' + str(num_items))
    print('poolsize: ' + str(poolsize))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    with open('../data/cifar100/order', 'r') as f:
        print('loading order...')
        file = f.read()
        order = eval(file)
    for i in range(num_users):
        idxs = np.random.choice(range(min(poolsize, len(order))), num_items, replace=False)
        dict_users[i] = [order[idx] for idx in idxs]
        order = [order[i] for i in range(len(order)) if i not in idxs]
    return dict_users


def cifar10_clorder(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    print('datasize of every client: ' + str(num_items))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    with open('../data/cifar/cifar10order', 'r') as f:
        file = f.read()
        order = eval(file)
    for i in range(num_users):
        dict_users[i] = order[i * num_items:(i + 1) * num_items]
    for k, v in dict_users.items():
        print(sum([order.index(i) for i in v]))
    return dict_users


def tinyimagenet_clorder(dataset, num_users):
    """
    Sample I.I.D. client data from tinyimagenet dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    print('datasize of every client: ' + str(num_items))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    with open('../data/tiny-imagenet-200/imageorderonres', 'r') as f:
        print('loading order...')
        file = f.read()
        order = eval(file)

    for i in range(num_users):
        dict_users[i] = order[i * num_items:(i + 1) * num_items]
        # dict_users[i] = order[i::num_users]
    return dict_users
