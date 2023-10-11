# coding = utf-8

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

# parameters
Batch_size = 32
Lr = 0.01
Gamma = 0.8  # reward discount
Target_replace_iter = 16  # target update frequency
Memory_capacity = 64
N_actions = 3
N_states = 110


class Net(nn.Module):
    def __init__(self, N_states=110, N_actions=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(512, N_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, path, N_states=110, N_actions=3):
        self.eval_net, self.target_net = Net(N_states, N_actions), Net(N_states, N_actions)
        if os.path.exists(path):
            self.eval_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(torch.load(path))
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))  # innitialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Lr)
        self.loss_func = nn.MSELoss()
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.path = path

    def update_epsilon(self):
        self.epsilon = self.epsilon - 0.7 / 100

    # action 0: -1; 1: +0; 2: +1
    def choose_action(self, x):
        # 把x变为[[1,2,...]]
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, N_actions)
            action = action
        else:
            print('take Qmax action.')
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        return action

    def get_topK(self, x, k):
        action_value = self.eval_net.forward(x)
        # action_value: [[...]]
        # torch.max(actions_value,1) 返回
        # values [max_v]
        # indices [max_v_index]
        return action_value.data.numpy()[0].argsort()[-k:]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, switch='on'):
        if self.memory_counter < Memory_capacity or switch == 'off':
            return

        # target net update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_states:N_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states + 1:N_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.eval_net.state_dict(), self.path)
