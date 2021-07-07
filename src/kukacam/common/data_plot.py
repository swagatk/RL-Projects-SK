'''
Statistical plots using seaborn and pandas
'''

import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

# define the problem here
PROB = 'racecar'

if PROB == 'fetch':
    # Performance Comparison:
    file1 = './results/freach_ppo_100k.txt'
    file2 = './results/freach_ipg_100k.txt'
    file3 = './results/freach_ipg_her.txt'
    file4 = './results/freach_ipg_her_attn.txt'

    df1 = pd.read_csv(file1, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df1["method"] = 'ppo'
    df2 = pd.read_csv(file2, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df2["method"] = 'ipg'
    df3 = pd.read_csv(file3, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df3["method"] = 'ipg_her'
    df4 = pd.read_csv(file4, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df4["method"] = 'ipg_her_attn'

    # join the dataframes
    df = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)

    plot_title = 'FetchReach-v1'

elif PROB == 'attn':    # Attention Ablation
    file1 = './results/luong_arch_1.txt'
    file2 = './results/bahdanau_arch_1.txt'
    file3 = './results/bahdanau_arch_2.txt'
    file4 = './results/bahdanau_arch_3.txt'
    file5 = './results/bahdanau_arch_4.txt'
    file6 = './results/luong_arch_32.txt'
    file7 = './results/luong_arch_31.txt'
    file8 = './results/luong_arch_21.txt'

    df1 = pd.read_csv(file1, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df1["method"] = 'Luong Arch 1'
    df2 = pd.read_csv(file2, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df2["method"] = 'Bahdanau Arch 1'
    df3 = pd.read_csv(file3, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df3["method"] = 'Bahdanau Arch 2'
    df4 = pd.read_csv(file8, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df4["method"] = 'Luong Arch 2'
    df5 = pd.read_csv(file4, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df5["method"] = 'Bahdanau Arch 3'
    df6 = pd.read_csv(file7, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df6["method"] = 'Luong Arch 3'
    df7 = pd.read_csv(file5, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df7["method"] = 'Bahdanau Arch 4'
    df8 = pd.read_csv(file6, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df8["method"] = 'Luong Arch 3_2'

    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8]).reset_index(drop=True)

    plot_title = 'KukaDiverseObject'

elif PROB == 'kuka':    # performance comparison for Kuka
    file1 = '../results/result_ppo.txt'
    file2 = '../results/result_ipg.txt'
    file3 = '../results/ipg_her.txt'
    file4 = '../results/ipg_her_attn.txt'

    df1 = pd.read_csv(file1, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df1["method"] = 'ppo'
    df2 = pd.read_csv(file2, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df2["method"] = 'ipg'
    df3 = pd.read_csv(file3, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df3["method"] = 'ipg_her'
    df4 = pd.read_csv(file4, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df4["method"] = 'ipg_her_attn'

    df = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)

    plot_title = 'KukaDiverseObject'

elif PROB == 'racecar':
    file2 = './results/rc_ipg.txt'
    file3 = './results/rc_ipg_her.txt'
    file4 = './results/rc_ipg_her_attn.txt'

    df2 = pd.read_csv(file2, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df2["method"] = 'ipg'
    df3 = pd.read_csv(file3, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df3["method"] = 'ipg_her'
    df4 = pd.read_csv(file4, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
    df4["method"] = 'ipg_her_attn'

    df = pd.concat([df2, df3, df4]).reset_index(drop=True)
    plot_title = 'racecarZEDGymEnv'
else:
    print('Nothing to do.')
    raise ValueError('Choose the right problem')

print(df.head())
print(df.tail())
print(df.shape)
trunc_df = df[['season', 'a_loss', 'c_loss', 'method']]
print(trunc_df.head())
sb.set_theme()
sb.set_style('whitegrid')
g1 = sb.relplot(x='season', y='mean_score', hue='method', kind='line', data=df)
# g1._legend.set_bbox_to_anchor([0.9, 0.3])
g1.set(title=plot_title)
trunc_df_melted = trunc_df.melt(id_vars=['season', 'method'], value_vars=['a_loss', 'c_loss'],\
                                var_name='loss_type', value_name='loss_value')
g2 = sb.relplot(x='season', y='loss_value', hue='loss_type', style='method', kind='line', data=trunc_df_melted)
g2.set(title=plot_title)
#g2._legend.set_bbox_to_anchor([0.9, 0.8])
plt.show()


