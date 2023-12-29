import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from agent_utils import (greedy_select_action, sample_select_action,
                         select_action)
from cloudedge_env import CloudEdge
from config import configs
from model.transformer import Encoder1

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

"""This part covers the two agents proposed in the article (Task Selection Agent and Computing Node Selection Agent"""


class task_actor(nn.Module):
    """
    Task Selection Agent;
    Output an action and the probability corresponding to the action at each scheduling step.
    """

    def __init__(self,
                 batch,
                 hidden_dim,
                 M):
        super().__init__()
        self.M = M

        self.hidden_dim = hidden_dim

        self.task_encoder = Encoder1(Inputdim=configs.input_dim1,
                                     embedding_size=configs.hidden_dim,
                                     M=M).to(DEVICE)

        self.batch = batch

        self.wq = nn.Linear(hidden_dim, hidden_dim)

        self.wk = nn.Linear(hidden_dim, hidden_dim)

        self.wv = nn.Linear(hidden_dim, hidden_dim)

        self.w = nn.Linear(hidden_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)

        self.k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data, index, feasibility, mask, action_probability, train):

        mask = torch.from_numpy(mask).to(DEVICE)

        tag = torch.LongTensor(self.batch).to(DEVICE)
        for i in range(self.batch):
            tag[i] = configs.n_j * i

        C = 10

        nodes, grapha = self.task_encoder(feasibility)

        torch.cuda.empty_cache()

        q = grapha

        dk = self.hidden_dim / self.M

        query = self.wq(q)  # (batch, embedding_size)

        query = torch.unsqueeze(query, dim=1)

        query = query.expand(self.batch, configs.n_j, self.hidden_dim)

        key = self.wk(nodes)

        temp = query * key

        temp = torch.sum(temp, dim=2)

        temp = temp / (dk ** 0.5)

        temp = torch.tanh(temp) * C

        temp.masked_fill_(mask, float('-inf'))

        p = F.softmax(temp, dim=1)  #

        ppp = p.view(1, -1).squeeze()

        p = torch.unsqueeze(p, dim=2)

        if train == 1:
            action_index = select_action(p)
        elif train == 2:
            action_index = sample_select_action(p)
        else:
            action_index = greedy_select_action(p)

        action_probability[tag + index] = ppp[tag + action_index]  # wenti

        dur_edge_execution = np.array(data[2], dtype=np.single)  # single  ##

        dur_cloud_execution = np.array(data[3], dtype=np.single)

        dur_sending = np.array(data[4], dtype=np.single)

        datasize = np.array(data[0], dtype=np.single)

        T = np.array(data[1], dtype=np.single)

        process_time = np.zeros((self.batch, 2), dtype=np.single)

        for i in range(self.batch):

            process_time[i][0] = dur_edge_execution[i][action_index[i]]

            process_time[i][1] = dur_cloud_execution[i][action_index[i]]  # 这里为何不加dur_sending?

        return action_index, action_probability, process_time  # (batch,1)


class place_actor(nn.Module):
    """Computing Node Selection Agent
        Output an action and the probability corresponding to the action at each scheduling step"""

    def __init__(self, batch, hidden_dim, M,):
        super().__init__()

        self.M = M

        self.hidden_dim = hidden_dim

        self.p_encoder = Encoder1(Inputdim=configs.input_dim2,
                                  embedding_size=hidden_dim,
                                  M=M).to(DEVICE)

        self.batch = batch

        self.wq = nn.Linear(hidden_dim, hidden_dim)

        self.wk = nn.Linear(hidden_dim, hidden_dim)

        self.wv = nn.Linear(hidden_dim, hidden_dim)

        self.w = nn.Linear(hidden_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)

        self.k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, index, task_op, place_action_probability, place_time, process_time, train):

        p_feas = np.concatenate((place_time.reshape(self.batch, 2, 1),
                                process_time.reshape(self.batch, 2, 1)), axis=2)

        C = 10

        p_tag = torch.LongTensor(self.batch).to(DEVICE)

        for i in range(self.batch):
            p_tag[i] = configs.n_j * i
        p_tag1 = torch.LongTensor(self.batch).to(DEVICE)
        for i in range(self.batch):
            p_tag1[i] = 2 * i

        nodes, grapha = self.p_encoder(p_feas)

        torch.cuda.empty_cache()

        q = grapha

        dk = self.hidden_dim / self.M

        query = self.wq(q)  # (batch, embedding_size)

        query = torch.unsqueeze(query, dim=1)

        query = query.expand(self.batch, 2, self.hidden_dim)

        key = self.wk(nodes)

        temp = query * key

        temp = torch.sum(temp, dim=2)

        temp = temp / (dk ** 0.5)

        temp = torch.tanh(temp) * C

        p = F.softmax(temp, dim=1)  #

        ppp = p.view(1, -1).squeeze()

        p = torch.unsqueeze(p, dim=2)

        if train == 1:
            action_index = select_action(p)
        elif train == 2:
            action_index = sample_select_action(p)
        else:
            action_index = greedy_select_action(p)

        place_action_probability[p_tag + index] = ppp[p_tag1 + action_index]

        return action_index, place_action_probability


class actor_critic(nn.Module):
    """Two agents work together to obtain scheduling results"""

    def __init__(self,
                 batch,
                 hidden_dim,
                 M,
                 device):

        super().__init__()

        self.M = M

        self.hidden_dim = hidden_dim

        self.env = CloudEdge(n_j=configs.n_j, maxtasks=configs.maxtask, max_Mem=configs.Mem)

        self.actor1 = task_actor(batch=batch, hidden_dim=hidden_dim, M=M)

        self.actor2 = place_actor(batch=batch, hidden_dim=hidden_dim, M=M)

        self.batch = batch

    def forward(self, data, train):

        action_probability = torch.zeros(self.batch, configs.n_j).to(DEVICE).view(1, -1).squeeze().to(DEVICE)

        place_action_probability = torch.zeros(self.batch, configs.n_j).to(DEVICE).view(1, -1).squeeze().to(DEVICE)

        task_feasibility, task_mask, place_time = self.env.reset(self.batch, data)

        task_seq_list = []

        place_operation_list = []

        rewards = 0

        q = torch.zeros((self.batch, 1)).to(DEVICE)

        for i in range(configs.n_j):

            index = i

            task_operation, action_probability, process_time = self.actor1(data, index, task_feasibility, task_mask, action_probability, train)  # 选择任务

            ind = torch.unsqueeze(task_operation, 1).tolist()

            task_seq_list.append(task_operation)

            place_operation, place_action_probability = self.actor2(index, task_operation, place_action_probability, place_time, process_time, train)

            place_operation_list.append(place_operation)

            task_feasibility, task_mask, place_time, reward = self.env.step(task_operation, place_operation)

            rewards += reward

        place_action_probability = place_action_probability.view(self.batch, configs.n_j)  # (batch,n)

        task_action_probability = action_probability.view(self.batch, configs.n_j)  # (batch,n)！！！！！

        task_seq = torch.unsqueeze(task_seq_list[0], 1)

        place_seq = torch.unsqueeze(place_operation_list[0], 1)

        q = q.to(DEVICE)

        for i in range(configs.n_j - 1):
            task_seq = torch.cat([task_seq, torch.unsqueeze(task_seq_list[i + 1], 1)], dim=1)
        for i in range(configs.n_j - 1):
            place_seq = torch.cat([place_seq, torch.unsqueeze(place_operation_list[i + 1], 1)], dim=1)
        # print(task_seq[0])

        rewards = torch.from_numpy(rewards).to(DEVICE)

        rewards = rewards.to(torch.float32)

        return task_seq, place_seq, task_action_probability, place_action_probability, rewards

    def update(self, task_action_pro, reward1, q, lr):

        opt = optim.Adam(self.actor1.parameters(), lr)

        pro = torch.log(task_action_pro)

        loss = torch.sum(pro, dim=1)

        score = reward1 - q

        score = score.detach()

        loss = score * loss

        loss = torch.sum(loss) / configs.batch

        opt.zero_grad()
        loss.backward()
        opt.step()

    def update2(self, place_action_pro, reward1, q, lr):

        opt = optim.Adam(self.actor2.parameters(), lr)

        pro = torch.log(place_action_pro)

        loss = torch.sum(pro, dim=1).to(DEVICE)

        score = reward1 - q

        score = score.detach().to(DEVICE)

        loss = score * loss

        loss = torch.sum(loss) / configs.batch

        opt.zero_grad()
        loss.backward()
        opt.step()
