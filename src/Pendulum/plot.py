import numpy as np
import matplotlib.pyplot as plt


def readFromFile(fname):
    mat = []
    file = open(fname, 'r')
    for line in file:
        fields = line.split("\t")
        cols = [float(s) for s in fields]
        mat.append(cols)
    return np.array(mat)


data1 = readFromFile('./results/result_clip_1.txt')
data2 = readFromFile('./results/result_klp_1.txt')
data3 = readFromFile('./results/result_clip_2.txt')
data4 = readFromFile('./results/result_klp_2.txt')

print(np.shape(data1))

plt.plot(data1[:, 0]*50, data1[:, 1], 'r-', lw=2, label='PPO-Clip-1')
plt.plot(data2[:, 0]*50, data2[:, 1], 'b-', lw=2, label='KL-Penalty-1')
plt.plot(data3[:, 0], data3[:, 2], 'm-', lw=2, label='PPO-Clip-2')
plt.plot(data4[:, 0], data4[:, 2], 'g-', lw=2, label='KL-Penalty-2')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.legend(loc='best')
plt.grid()
plt.savefig('./images/ppo-pendulum-reward-12.png')
plt.show()
