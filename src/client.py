#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset


# dataset[idxs] -> subdataset on client
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
        return image.clone().detach(), torch.tensor(label)
        # return torch.tensor(image), torch.tensor(label)


# data_idxs: index in the dataset
class Client(object):
    def __init__(self, args, dataset, data_idxs, global_round):
        self.args = args
        self.trainloader, self.train_subsetloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(data_idxs), global_round)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_num = len(self.trainloader)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, data_idxs, global_round):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        percent = 1
        idxs_train = data_idxs[:int(percent * len(data_idxs))]
        idxs_val = data_idxs[int(0.8 * len(data_idxs)):int(1 * len(data_idxs))]
        idxs_test = data_idxs[int(0.9 * len(data_idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, num_workers=2, pin_memory=True)
        train_subsetloader = DataLoader(
            DatasetSplit(dataset, np.random.choice(idxs_train, int(0.1 * len(data_idxs)), replace=False)),
            batch_size=self.args.local_bs, shuffle=True, num_workers=2, pin_memory=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, shuffle=True, num_workers=2, pin_memory=True)
        return trainloader, train_subsetloader, validloader, None

    def update_weights(self, model, interacts_list):
        # Set mode to train model
        model.to(self.device)
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum, weight_decay=self.args.wd)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # print('batch num: {:d}'.format(len(self.trainloader)))
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                # print(log_probs) two dimensions
                # print(labels) one dimensions
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_local_loss(self, model):

        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            log_probs = model(images)
            loss = self.criterion(log_probs, labels)

            batch_loss.append(loss.item())

        return sum(batch_loss) / len(batch_loss)

    # @para train_subset or validset
    def inference(self, model, dataset_sel='train_subset'):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if dataset_sel == 'train_subset':
            dataloader = self.train_subsetloader
        else:
            dataloader = self.validloader

        batch_losses = []
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            # loss += batch_loss.item()
            batch_losses.append(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        loss = sum(batch_losses) / len(batch_losses)
        return accuracy, loss

    def update_lr(self, interacts_list):
        if sum(interacts_list) < 2200:
            return self.args.lr
        else:
            return self.args.lr * 0.1
