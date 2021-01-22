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


data1 = readFromFile('./results/result_clip.txt')
data2 = readFromFile('./results/result_penalty2.txt')
#data3 = readFromFile('./results/result_penalty2.txt')

print(np.shape(data1))

#plt.plot(data1[:, 0], data1[:, 4], 'r-', label='PPO-Clip')
plt.plot(data2[:, 0], data2[:, 5], 'b-', label='PPO-KL-Penalty')
#plt.plot(data3[:, 0], data3[:, 1], 'g-', label='Dueling_DQN')
plt.xlabel('Episodes')
plt.ylabel('KL Divergence')
plt.legend(loc='best')
plt.grid()
plt.savefig('./images/ppo-pendulum-kld.png')
plt.show()
