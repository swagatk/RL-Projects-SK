# Reading data files and plotting figures

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
            
#data1 = readFromFile('./data2/cp_dqn_bs24.txt')
#data2 = readFromFile('./data2/cp_ddqn_bs24.txt')
#data3 = readFromFile('./data2/cp_ddqn_PA_bs24.txt')
#data4 = readFromFile('./data2/cp_dqn_PER_bs24.txt')
#data5 = readFromFile('./data2/cp_ddqn_PER_bs24.txt')

#data1 = readFromFile('./data2/cp_d3qn_bs24.txt')
#data2 = readFromFile('./data2/cp_d3qn_PER_bs24.txt')

data1 = readFromFile('./data2/cp_d3qn_bs32_512:256:64.txt')
data2 = readFromFile('./data2/cp_d3qn_PER_bs32_512:256:64.txt')
data3 = readFromFile('./data2/cp_d3qn_PER_PA_bs32:512:256:64.txt')


print(np.shape(data1))

plt.plot(data1[:,0], data1[:,2],'r-',label='D3QN')
plt.plot(data2[:,0], data2[:,2],'b-', label='D3QN w PER')
plt.plot(data3[:,0], data3[:,2],'g-', label='D3QN w PER and PA')

#plt.plot(data1[:,0], data1[:,3],'r-',label='DQN')
#plt.plot(data2[:,0], data2[:,3],'b-', label='DDQN')
#plt.plot(data3[:,0], data3[:,3],'g-', label='DDQN w PA')
#plt.plot(data4[:,0], data4[:,3],'m-', label='DQN w PER')
#plt.plot(data5[:,0], data5[:,3],'c-', label='DDQN w PER')

plt.xlabel('Episodes')
plt.ylabel('Average Scores')
plt.legend(loc='lower right')
#plt.savefig('./img/cp_D3QN_PER_avg100:512:256:64.png')
plt.savefig('./img/cp_D3QN_PER_PA_avg:512:256:64.png')
#plt.savefig('./img/cp_DQN_PER:24:24:2.png')
plt.show()
