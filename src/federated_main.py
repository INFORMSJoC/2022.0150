import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
import math

import torchvision

from options import args_parser
from client import Client
from utils import get_dataset, average_weights, print_exp_details, test_inference, cal_state
from ddqn import DQN
from models import Wide_ResNet, ResNet18
from vgg import vgg16_bn
import newRes

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()

    print_exp_details(args)

    file_name = './save/{}_{}_method[{}]_{}.pkl'. \
        format(args.dataset, args.model, args.method, args.pacefunc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    dqn_agent = DQN('ddqnpara.pkl')

    if args.model == 'resnet18':
        global_model = ResNet18()
    elif args.model == 'resnext50':
        global_model = torchvision.models.resnext50_32x4d(pretrained=False)
        num_features = global_model.fc.in_features
        global_model.fc = torch.nn.Sequential(torch.nn.Linear(num_features, 100))
    elif args.model == 'WRN':
        global_model = Wide_ResNet(28, 10, 0.3, 100)
    elif args.model == 'vgg':
        global_model = vgg16_bn()
    elif args.model == 'resnet50':
        global_model = newRes.ModelFedCon()
    else:
        print("do not support model: %s" % args.model)
        raise ValueError

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    # the number of selected clients in each round, subject to [S_0, S_1, ..., S_r, ...]
    interacts_list = []
    # Test
    test_loss, test_accuracy = [], []

    random_users_rank = [i for i in range(args.num_users)]
    random.shuffle(random_users_rank)
    pre_s = None
    pre_action = None
    pre_r = None

    for round_r in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round_r + 1} |\n')

        global_model.train()

        # percent: M_r / K
        if args.pacefunc == 'const':
            percent = 1
        elif args.pacefunc == 'linear1':
            percent = min(100.0, int(30 + round_r * 0.2)) / 100
        elif args.pacefunc == 'linear2':
            percent = min(100.0, int(30 + round_r * 0.49)) / 100
        elif args.pacefunc == 'decrease':
            percent = max(1 / math.exp(round_r * 0.0033743), 0.3)
        elif args.pacefunc == 'exp':
            percent = min(100.0, (30 + math.exp(round_r / 83) - 1)) / 100
        elif args.pacefunc == 'step':
            percent = min(100.0, (30 + int(round_r / 180) * 70)) / 100
        elif args.pacefunc == 'step1':
            percent = min(100.0, (100 - int(round_r / 250) * 70)) / 100
        elif args.pacefunc == 'sin':
            if round_r < 260:
                percent = min(1.0, 0.3 + 0.7 * math.sin(round_r * math.pi / 520))
            else:
                percent = 1.0
        elif args.pacefunc == 'drl':
            if round_r == 0:
                percent = 0.3
            else:
                percent = max(min(interacts_list[-1] / 10, 1), 0.3)
        else:
            print("do not support pacefunc: %s" % args.pacefunc)
            raise ValueError

        if args.method in ['fedecs', 'ascend', 'afl']:
            m_r = int(args.num_users * percent * args.frac)
            if args.pacefunc == 'drl':
                m_r = int(args.num_users * percent * args.frac) + 1
                m_r = max(min(m_r, 10), 3)
        elif args.method in ['fedavg', 'favor']:
            percent = 1
            m_r = int(args.num_users * args.frac * percent)
        else:
            print("do not support method: %s" % args.method)
            raise ValueError

        transfer_users_rank = [i for i in user_groups.keys()]

        if args.selfrank and round_r % 10 == 0:
            temp_local_losses = []
            for idx in range(args.num_users):
                local_model = Client(args=args, dataset=train_dataset,
                                     data_idxs=user_groups[idx], global_round=round_r)
                _, local_loss = local_model.inference(model=global_model)
                temp_local_losses.append(local_loss)

            self_users_rank = [i for i in range(args.num_users)]
            Z = zip(temp_local_losses, self_users_rank)
            Z = sorted(Z, reverse=False)
            temp_local_losses, self_users_rank = zip(*Z)

        # get the indices of the selected clients in round r, subject to S_r
        # available_users: M_r, selected_user: S_r
        if args.method == 'fedecs':
            available_users = transfer_users_rank[0:int(args.num_users * percent)]
            selected_users = np.random.choice(available_users, m_r, replace=False)
            print(available_users)
        elif args.method == 'ascend':
            available_users = random_users_rank[0:int(args.num_users * percent)]
            selected_users = np.random.choice(available_users, m_r, replace=False)
            print(available_users)
        elif args.method == 'afl':
            available_users = transfer_users_rank[-int(args.num_users * percent):]
            selected_users = np.random.choice(available_users, m_r, replace=False)
            print(available_users)
        elif args.method == 'favor':
            percent = min(100.0, int(30 + round_r * 70 / (0.7143*args.interacts/10))) / 100
            available_users = transfer_users_rank[0:int(args.num_users * percent)]
            selected_users = np.random.choice(available_users, m_r, replace=False)
            print(available_users)
        else:
            selected_users = np.random.choice(range(args.num_users), m_r, replace=False)

        print(selected_users)
        interacts_list.append(len(selected_users))
        for idx in selected_users:
            local_model = Client(args=args, dataset=train_dataset,
                                 data_idxs=user_groups[idx], global_round=round_r)
            # tmp_model = vgg16_bn()
            # tmp_model = newRes.ModelFedCon()
            # tmp_model.load_state_dict(global_model.state_dict())
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), interacts_list=interacts_list)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if args.verbose:
                print('| Global Round : {:3d} | Trained Client : {:3d} | \tLoss: {:.6f} | \tBatchNum: {}'.format(
                    round_r, idx, loss, local_model.batch_num))

        if args.pacefunc == 'drl':
            selected_users_p0 = np.random.choice(range(len(selected_users)), m_r - 2, replace=False)
            selected_users_p1 = np.random.choice(range(len(selected_users)), m_r - 1, replace=False)
            selected_users_p2 = range(len(selected_users))
            global_weights_p0 = average_weights([local_weights[i] for i in selected_users_p0])
            global_weights_p1 = average_weights([local_weights[i] for i in selected_users_p1])
            global_weights_p2 = average_weights([local_weights[i] for i in selected_users_p2])

            s = cal_state(local_weights, round_r)

            if pre_s is not None:
                dqn_agent.store_transition(pre_s, pre_action, pre_r, s)

            if round_r < 3:
                action = 1
            else:
                # action 0: |S_r|-1; 1: |S_r|+0; 2: |S_r|+1
                action = dqn_agent.choose_action(s)
                # cal reward
                global_model.eval()
                list_acc, list_loss = [], []
                for idx in selected_users:
                    local_model = Client(args=args, dataset=train_dataset,
                                         data_idxs=user_groups[idx], global_round=round_r)
                    acc, loss = local_model.inference(model=global_model, dataset_sel='valid')
                    list_acc.append(acc)
                    list_loss.append(loss)
                reward = sum(list_acc) / len(list_acc) / (sum(interacts_list) - 1 + action)

                pre_s = s
                pre_action = action
                pre_r = reward

                dqn_agent.learn()

            selected_users = [selected_users[i] for i in
                              [selected_users_p0, selected_users_p1, selected_users_p2][action]]
            global_weights = [global_weights_p0, global_weights_p1, global_weights_p2][action]
            global_model.train()
            global_model.load_state_dict(global_weights)
            interacts_list[-1] = len(selected_users)
            print(selected_users)
        else:
            # update global weights
            global_weights = average_weights(local_weights)
            global_model.train()
            # update global weights
            global_model.load_state_dict(global_weights)

        # cal train loss & acc of the selected clients
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        list_acc, list_loss = [], []
        global_model.eval()
        for idx in selected_users:
            local_model = Client(args=args, dataset=train_dataset,
                                 data_idxs=user_groups[idx], global_round=round_r)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        print(f' \nAvg Training Stats after {round_r + 1} global rounds:')
        print(f'Training Loss : {train_loss[-1]}')
        print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        # test on test_dataset
        acc, loss = test_inference(args, global_model, test_dataset)
        test_accuracy.append(acc), test_loss.append(loss)
        print(
            "|---- Test Accuracy: {:.2f}% after {} interactions.".format(100 * test_accuracy[-1], sum(interacts_list)))

        # check stopping by interacts number
        if sum(interacts_list) > args.interacts:
            break

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy, test_accuracy, interacts_list], f)

    print(f' \n Results after {args.rounds} global rounds of training:')
    print("|---- Final Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Final Test Accuracy: {:.2f}%".format(100 * test_accuracy[-1]))

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, test_accuracy, interacts_list], f)
        print('log has been saved in ' + file_name)

    dqn_agent.save_model()

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    print_exp_details(args)
