import re

import matplotlib.pyplot as plt
import numpy as np

from config import configs

"""Draw the convergence curve of the model based on the data saved during the training process"""


def plot_loss():

    compare = 1

    max = 8

    f = open('.//train_vali//{}//compare{}//{}_{}.txt'.format(configs.n_j, compare, configs.n_j, max), 'r')

    data = []

    for line in f:
        data.append(re.sub('\D', '', line))

    print(data)

    datas = []

    for i in data:
        datas.append(int(i) / 100000)

    f.close()

    print(datas)

    fig, ax = plt.subplots()

    plt.rc('font', family='Times New Roman', size=12)  # Times New Roman

    x = np.arange(1, len(datas) + 1)

    y = datas

    plt.plot(x, y, c='dimgray', marker='.', ms=0, label="a")

    plt.xlabel("4 Parallel Computing Capability Training Step(*20)", fontdict={'family': 'Times New Roman', 'size': 18})

    plt.ylabel("Time Delay", fontdict={'family': 'Times New Roman', 'size': 18})

    plt.yticks(fontproperties='Times New Roman', size=12)

    plt.xticks(fontproperties='Times New Roman', size=12)

    ax.set_facecolor("whitesmoke")  # backgrand color

    plt.legend(['Loss Function'], prop={'family': 'Times New Roman', 'size': 14})

    plt.grid(True, alpha=0.5)  # grand

    # ax.spines['top'].set_visible(False)  # remove up

    # ax.spines['right'].set_visible(False)  # remove down

    plt.savefig("20_4p.jpg", dpi=600)

    plt.show()


def show_data():
    data = np.load(r"data2\10\compare1\datas10_1000_2000.npy")
    print(data[1][1])


if __name__ == '__main__':
    plot_loss()
