import os
import copy
import time
import pickle
import numpy as np
from torch._C import dtype
from tqdm import tqdm
import random

import torch
import math

import torchvision

from options import args_parser
from client import LocalModel
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

    file_name = './save/{}_{}_CLpara[{}]_lr[{}]_{}.pkl'. \
        format(args.dataset, args.model, args.CLpara, args.lr, args.pacefunc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    # user groups  {user_id: [data_idxs in dataset]}
    train_dataset, test_dataset, user_groups = get_dataset(args)

    dqn_agent = DQN('ddqnpara.pkl', N_states=250, N_actions=250)
    i_episode = 100

    for i in range(i_episode):

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

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        # the number of selected clients in each round
        interacts_list = []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0
        # Test
        test_loss, test_accuracy = [], []

        random_users_idxs = [i for i in range(args.num_users)]
        random.shuffle(random_users_idxs)
        w_fc_l2_norm = 0
        w_fc_l2_norm_0 = 0
        pre_s=None
        pre_action=None
        pre_r=None

        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            # print(f'\n | Global Training Round : {epoch + 1} |\n')

            global_model.train()

            m = 10 

            temp_local_losses = []
            for idx in range(args.num_users):
                local_model = LocalModel(args=args, dataset=train_dataset,
                                        data_idxs=user_groups[idx], global_epoch=epoch)
                _, local_loss = local_model.inference(model=global_model, is_train=1)
                temp_local_losses.append(local_loss)

            idxs_users_rank = [i for i in range(args.num_users)]
            Z = zip(temp_local_losses, idxs_users_rank)
            
            Z = sorted(Z, reverse=False)
            temp_local_losses, idxs_users_rank = zip(*Z)

            # FAVOR adopts all model weights as the state, which needs PCA to reduce dimensions and thus costs computation.
            # To simplify it, we use loss of each client as the state based on some heuristics.
            s = np.asarray(temp_local_losses, dtype=float)

            if pre_s is not None:
                dqn_agent.store_transition(pre_s,pre_action,pre_r,s)

            action = dqn_agent.choose_action(s)
            user_id = action
            print(user_id)

            local_model = LocalModel(args=args, dataset=train_dataset,
                                     data_idxs=user_groups[user_id], global_epoch=epoch)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), interacts_list=interacts_list)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if args.verbose:
                print('| Global Round : {:3d} | Trained Client : {:3d} | \tLoss: {:.6f} | \tBatchNum: {}'.format(
                    epoch, user_id, loss, local_model.batch_num))

            global_weights, w_fc_l2_norm = average_weights(local_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            # cal reward
            global_model.eval()
            list_acc, list_loss = [], []
            for idx in range(args.num_users):
                local_model = LocalModel(args=args, dataset=train_dataset,
                                        data_idxs=user_groups[idx], global_epoch=epoch)
                acc, loss = local_model.inference(model=global_model,is_train=0)
                list_acc.append(acc)
                list_loss.append(loss)
            r = sum(list_acc)/len(list_acc)

            pre_s=s
            pre_action=action
            pre_r=r

            dqn_agent.learn()

            if epoch % 10 ==0:
                acc, loss = test_inference(args, global_model, test_dataset)
                print(
            "|---- Test Accuracy: {:.2f}% after {} epochs.".format(100 * acc, epoch))
                if acc > 0.4:
                    break
    
        dqn_agent.save_model()
        dqn_agent.update_epsilon()


    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    print_exp_details(args)
