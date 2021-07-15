import os
import matplotlib.pyplot as plt
import pandas as pd

root_path = '/content/gdrive/My Drive/Colab/SB3/kuka/'
monitor_path = root_path + 'monitor/'

# read CSV file while skipping the first row and using the second row as headers
# don't use the first column as index.
df = pd.read_csv(monitor_path+'monitor.csv', skiprows=0, header=1, index_col=None)

print('datatype:', type(df))
print(df.shape)
# data preview
print(df.head())

print(df.columns)

# cumulative sum
print(df['r'].cumsum(axis=0))

df['cumr'] = df['r'].cumsum(axis=0)
df['ep_r'] = df['cumr'] / (df.index + 1)

print(df.head())

total_episodes = df.shape[0]
print('-----------------------')
print('Total Number of episodes: ', total_episodes)
print('Average Episodic Reward:', df['r'].sum(axis=0) / total_episodes)
print('Mean Episode Length:', df['l'].mean())
print('----------------------')
plt.figure(0)
g1 = df['l'].plot()
g1.set_xlabel('Episodes')
g1.set_ylabel('Episode Length')
plt.savefig('kuka_ep_len.png')

plt.figure(1)
g2 = df['ep_r'].plot(linewidth=3, color='red')
g2.set_xlabel('Episodes')
g2.set_ylabel('Mean Episodic Reward')
# plt.plot(x = 'Index', y='ep_r', data=df, color='red', linewidth=2)
plt.savefig('kuka_ep_reward.png')
plt.show()