#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.transforms import Resize
from sampling import cifar100_partdata
from sampling import cifar_iid, cifar100_iid, cifar100_clorder, \
    cifar10_clorder, tinyimagenet_clorder
import random
import numpy as np
import os
from PIL import Image
import pickle


class TinyImageNet(Dataset):
    def __init__(self, transform=None, is_train=True):
        self.data_dir = r"../data/tiny-imagenet-200/"
        with open(self.data_dir + 'wnids.txt', 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
        label_to_wnid = {v: k for k, v in wnid_to_label.items()}

        self.img_files = []
        self.img_labels = []
        if is_train:
            for k, v in wnid_to_label.items():
                images_path = self.data_dir + 'train' + '/' + str(k) + '/images/'
                images_name = os.listdir(images_path)
                for name in images_name:
                    self.img_files.append(images_path + name)
                    self.img_labels.append(v)
        else:
            with open(os.path.join(self.data_dir, 'val', 'val_annotations.txt'), 'r') as f:
                img_files = []
                val_wnids = []
                for line in f:
                    img_file, wnid = line.split('\t')[:2]
                    img_files.append(img_file)
                    val_wnids.append(wnid)
                self.img_files = [self.data_dir + 'val/images/' + item for item in img_files]
                self.img_labels = [wnid_to_label[wnid] for wnid in val_wnids]

        self.transform = transform

    def __getitem__(self, index):
        # 读取图像数据并返回
        # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
        with open(self.img_files[index], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            # img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.img_labels[index]

    def __len__(self):
        # 返回图像的数量
        return len(self.img_files)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar10':
        data_dir = '../data/cifar/'
        apply_transform1 = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], inplace=True)
        ])

        apply_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], inplace=True)
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform1)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform2)

        # cl order
        user_groups = cifar10_clorder(train_dataset, args.num_users)
        # cl order

    elif args.dataset == 'cifar100':
        data_dir = '../data/cifar100/'
        apply_transform1 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15), 6.15
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        apply_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225], inplace=True)
             ])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=apply_transform1)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                         transform=apply_transform2)

        # cl order
        user_groups = cifar100_clorder(train_dataset, args.num_users)
        # cl order

        # alpha order
        # user_groups = cifar100_partdata(train_dataset, args.num_users, args.alpha)
        # alpha order

    elif args.dataset == 'TinyImageNet':
        apply_transform1 = transforms.Compose([
            # for resnet50
            transforms.RandomResizedCrop(64),
            # for vgg
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        apply_transform2 = transforms.Compose([
            # for resnet50
            transforms.Resize(64),
            # for vgg
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225], inplace=True)
             ])

        train_dataset = TinyImageNet(transform=apply_transform1, is_train=True)

        test_dataset = TinyImageNet(transform=apply_transform2, is_train=False)

        # cl order
        user_groups = tinyimagenet_clorder(train_dataset, args.num_users)
        # cl order

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
        # w_avg[key] = torch.true_divide(w_avg[key], len(w))
    return w_avg


def cal_state(w, round):
    fc_w_avg = copy.deepcopy(w[0])['linear.weight']
    for i in range(1, len(w)):
        fc_w_avg += w[i]['linear.weight']
    fc_w_avg = torch.div(fc_w_avg, len(w))
    # fc_w_avg = torch.true_divide(fc_w_avg,len(w))
    fc_w_avg = fc_w_avg.cpu().numpy()
    # len(w)*100*640 to 100

    w_fc_norm = np.zeros((100, 640), dtype=float)
    for i in range(0, len(w)):
        fc_minus_square = np.square(w[i]['linear.weight'].cpu().numpy() - fc_w_avg)
        w_fc_norm += fc_minus_square

    s_round = np.asarray([round for i in range(5)], dtype=float)
    s_num_sel = np.asarray([len(w) for i in range(5)], dtype=float)
    s_w_fc_norm = np.average(w_fc_norm / len(w), axis=1)

    return np.concatenate((s_w_fc_norm, s_round, s_num_sel))


def print_exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    if args.optimizer == 'sgd':
        print(f'    Momentum : {args.momentum}')
    print(f'    Learning  : {args.lr}')
    print(f'    Weight_decay  : {args.wd}')
    print(f'    Global Rounds   : {args.rounds}')
    print(f'    Total Clients   : {args.num_users}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}\n')
    if args.selfrank:
        print(f'    selfrank')
    return


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.NLLLoss().to(device)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=20,
                                             shuffle=True, num_workers=2, pin_memory=True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction

        _, pred_labels = torch.max(outputs, 1)
        # pred_labels = pred_labels.view(-1)

        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss
