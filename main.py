import os

import numpy as np
import torch

from config import configs
from model.actor_critic import actor_critic

"""The main function of model training"""


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

compare = 1

size = '1000_2000'

"""Load training data and test data"""

datas = np.load('data2//{}//compare{}//datas{}_{}.npy'.format(configs.n_j, compare, configs.n_j, size))

datas.astype('float16')

print(datas.dtype)

testdatas = np.load('data2//{}//compare{}//com_testdatas{}_{}.npy'.format(configs.n_j, compare, configs.n_j, size))

Net1 = actor_critic(batch=configs.batch, hidden_dim=configs.hidden_dim, M=8, device=configs.device).to(DEVICE)

Net2 = actor_critic(batch=configs.batch, hidden_dim=configs.hidden_dim, M=8, device=configs.device).to(DEVICE)

Net2.place_actor.load_state_dict(Net1.place_actor.state_dict())

min = 50000000000


if configs.batch == 24:
    lr = 0.000001
    print('lr=', lr)

elif configs.batch == 8:
    lr = 0.0000001
    print('lr=', lr)

bl_alpha = 0.05

output_dir = 'train_process//{}//compare{}'.format(configs.n_j, compare)

save_dir = os.path.join(os.getcwd(), output_dir)

from scipy.stats import ttest_rel

contintrain = 0

Net2.load_state_dict(Net1.state_dict())

for epoch in range(configs.epochs):
    for i in range(configs.time):
        data = datas[i]
        # print(data.shape)

        task_seq, place_seq, task_action_probability, place_action_probability, reward_task = Net1(data, 1)

        _, _, _, _, reward_place = Net2(data, 1)

        reward_task = reward_task.detach()

        torch.cuda.empty_cache()

        Net1.update_task(task_action_probability, reward_task, reward_place, lr)

        Net1.update_place(place_action_probability, reward_task, reward_place, lr)

        print('epoch={},i={},time1={},time2={}'.format(epoch, i, torch.mean(reward_task), torch.mean(reward_place)))

        with torch.no_grad():

            if (reward_task.mean() - reward_place.mean()) < 0:

                # 如果小于0，则进行统计检验（t检验）和基线更新
                tt, pp = ttest_rel(reward_task.cpu().numpy(), reward_place.cpu().numpy())

                p_val = pp / 2

                assert tt < 0, "T-statistic should be negative"

                if p_val < bl_alpha:

                    print('Update baseline')

                    Net2.load_state_dict(Net1.state_dict())

            """Every 20 iterations check whether the model needs to save parameters"""

            if i % 20 == 0:

                length = torch.zeros(1).to(DEVICE)

                for j in range(configs.comtesttime):
                    torch.cuda.empty_cache()

                    _, _, _, _, r = Net1(testdatas[j], 0)

                    length = length + torch.mean(r)

                length = length / configs.comtesttime

                if length < min:
                    torch.save(Net1.state_dict(), os.path.join(save_dir, 'epoch{}-i{}-dis_{:.5f}.pt'.format(epoch, i, length.item())))

                    torch.save(Net1.state_dict(), os.path.join(save_dir, 'actor{}_mutil_actor.pt'.format(configs.n_j)))

                    min = length
                file_writing_obj1 = open('./train_vali/{}//compare{}//{}_{}.txt'.format(configs.n_j, compare, configs.n_j, configs.maxtask), 'a')

                file_writing_obj1.writelines(str(length) + '\n')

                print('length=', length.item(), 'min=', min.item())

                file_writing_obj1.close()
