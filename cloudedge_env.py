import gym
import numpy as np
import torch
from gym.utils import EzPickle

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

"""Environment for agent interaction, covering feature extraction and update"""


class CloudEdge(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 maxtasks,
                 max_Mem):
        EzPickle.__init__(self)
        self.maxtasks = maxtasks

        self.n_j = n_j

        self.maxMem = max_Mem

        self.step_count = 0

        self.L = 50  # penalty value

        self.number_of_jobs = n_j

        self.number_of_tasks_on_cloud = 0

    def reset(self, batch, data):
        """initialization"""
        self.batch = batch
        self.job_finish_time_cloud = np.zeros(self.batch * self.maxtasks).reshape((self.batch, -1))

        self.step_count = 0
        # print(self.step_count)

        # self.place = data[-1]

        self.dur_edge_execution = np.array(data[2], dtype=np.single)  # single

        self.dur_cloud_execution = np.array(data[3], dtype=np.single)

        self.dur_sending = np.array(data[4], dtype=np.single)

        self.datasize = np.array(data[0], dtype=np.single)

        self.T = np.array(data[1], dtype=np.single)
        # print('####',self.T.dtype)

        # task feature
        ##############################################################
        self.I = np.full(shape=(self.batch, self.n_j, 2), fill_value=0, dtype=bool)  # 这是一个存储任务是否已经被调度的掩码数组

        self.LBs = np.zeros((self.batch, self.n_j, 2), dtype=np.single)

        self.Fi = np.zeros((self.batch, self.n_j, 2), dtype=np.single)

        self.LBm = np.zeros((self.batch, self.n_j, 1), dtype=np.single)

        self.Fim = np.zeros((self.batch, self.n_j, 1), dtype=np.single)

        # self.G_LBs = np.ones((self.batch,self.n_j,2), dtype=np.single)
        self.place_time = np.zeros((self.batch, 2), dtype=np.single)

        self.task_mask = np.full(shape=self.T.shape, fill_value=0, dtype=bool)

        self.place_mask = np.full(shape=self.LBs.shape, fill_value=0, dtype=bool)
        # print('T',self.task_mask.shape)
        # self.Fi = np.zeros((self.batch,self.n_j,2), dtype=np.single)
        for i in range(self.batch):
            for j in range(self.n_j):
                # LBs[i][j][0]表示边缘任务完成总时间，LBs[i][j][1]表示云端任务完成总时间
                self.LBs[i][j][0] = self.dur_edge_execution[i][j]
                self.LBs[i][j][1] = self.dur_sending[i][j] + self.dur_cloud_execution[i][j]

                # Fi[i][j][0]和Fi[i][j][1]分别表示剩余的时间
                self.Fi[i][j][0] = self.T[i][j] - self.LBs[i][j][0]
                self.Fi[i][j][1] = self.T[i][j] - self.LBs[i][j][1]

                self.LBm[i][j][0] = min(self.LBs[i][j][0], self.LBs[i][j][1])

                self.Fim[i][j][0] = self.Fi[i][j][1]

        task_feasibility = np.concatenate((self.LBm.reshape(self.batch, self.n_j, 1),
                                           self.Fim.reshape(self.batch, self.n_j, 1),
                                           self.task_mask.reshape(self.batch, self.n_j, 1),

                                           ), axis=2)

        # print(self.I[0])
        return task_feasibility, self.task_mask, self.place_time

    def step(self, task_action, place_action):
        """Update features based on the actions of the agents"""
        for i in range(self.batch):
            if place_action[i] == 1:

                earliest_ind = np.argmin(self.job_finish_time_cloud[i])

                self.job_finish_time_cloud[i][earliest_ind] = self.LBs[i][task_action[i]][1]

                min_ind = np.argmin(self.job_finish_time_cloud[i])

                self.place_time[i][1] = self.job_finish_time_cloud[i][min_ind]

        reward = np.zeros((self.batch, 1))
        # print(self.job_finish_time_on_cloudy[0])

        for i in range(self.batch):
            if self.LBs[i][task_action[i]][place_action[i]] <= self.T[i][task_action[i]]:

                reward[i] = self.LBs[i][task_action[i]][place_action[i]]

            else:
                reward[i] = self.LBs[i][task_action[i]][place_action[i]] * 10
                # print('timewindows')

        # print(p_action[0])
        # print('reward',reward[0])
        earliest_time = np.zeros((self.batch, 1))
        for i in range(self.batch):
            earliest_time[i] = min(self.job_finish_time_cloud[i])
        # print(earlist_time[0])
        # print(place_time[0])

        for i in range(self.batch):
            self.I[i][task_action[i]][0] = True

            self.I[i][task_action[i]][1] = True

        for b in range(self.batch):
            self.task_mask[b][task_action[b]] = True  # 已调度任务mask

        for i in range(self.batch):
            for j in range(self.n_j):

                if self.I[i][j][0] == False and self.I[i][j][1] == False:
                    # EDGE
                    jobreadytime_edge = 0

                    compute_readytime_edge = 0

                    job_startime_edge = max(jobreadytime_edge, compute_readytime_edge)

                    job_finishtime_edge = job_startime_edge + self.dur_edge_execution[i][j]

                    self.LBs[i][j][0] = job_finishtime_edge

                    self.Fi[i][j][0] = self.T[i][j] - self.LBs[i][j][0]

                    # CLOUD
                    jobreadytime_cloud = self.dur_sending[i][j]

                    compute_readytime_cloud = min(self.job_finish_time_cloud[i])

                    job_startime_cloud = max(jobreadytime_cloud, compute_readytime_cloud)

                    job_finishtime_cloud = job_startime_cloud + self.dur_cloud_execution[i][j]

                    self.LBs[i][j][1] = job_finishtime_cloud

                    self.Fi[i][j][1] = self.T[i][j] - self.LBs[i][j][1]

                    self.LBm[i][j][0] = min(self.LBs[i][j][0], self.LBs[i][j][1])

                    self.Fim[i][j][0] = self.Fi[i][j][1]

        task_feasibility = np.concatenate((self.LBm.reshape(self.batch, self.n_j, 1),
                                           self.Fim.reshape(self.batch, self.n_j, 1),
                                           self.task_mask.reshape(self.batch, self.n_j, 1),
                                           ), axis=2)

        # print('LBs',self.LBs[0])
        # print('F',self.Fi[0])

        # print(self.task_mask[0])
        return task_feasibility, self.task_mask, self.place_time, reward


"""test"""
# env = CLOUD_edge(n_j=configs.n_j,
#                               maxtasks=configs.maxtask,
#                               max_Men=configs.Men)
# datas = np.load('data2//{}//compare{}//datas{}_{}.npy'.format(configs.n_j,1,configs.n_j,'1000_2000'))
# data = datas[0]
# task_feas,task_mask,place_time = env.reset(10, data)
# task = np.zeros((10,24),dtype=np.single)
# # task = task.float(32)
# task[0] = 0
# task[1],task[2],task[3],task[4],task[5],task[6],task[7],task[8],task[9] = 1,2,3,4,5,6,7,8,9
# task =task.astype(int)
# place = np.ones((10,24),dtype=np.single).astype(int)
# # print(task[0].reshape(24))
#
# dur_l = np.array(data[2], dtype=np.single)  # single  ##
# dur_e = np.array(data[3], dtype=np.single)
# dur_s = np.array(data[4], dtype=np.single)
# T = np.array(data[1], dtype=np.single)
# print('edge:',dur_l[0])
# print('trans',dur_s[0])
# print('cloud',dur_e[0])
# print('deadline',T[0])
# for i in range(10):
#
#     task_feas, task_mask, place_time, reward = env.step(task[i].reshape(24), place[i].reshape(24))
