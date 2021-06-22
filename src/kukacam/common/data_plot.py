'''
Statistical plots using seaborn and pandas
'''

import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

# Performance Comparison:
file3 = './results/freach_ipg_her.txt'
# Attention Ablation
# file1 = './results/luong_arch_1.txt'
# file2 = './results/bahdanau_arch_1.txt'
# file3 = './results/bahdanau_arch_2.txt'
# file4 = './results/bahdanau_arch_3.txt'
# file5 = './results/bahdanau_arch_4.txt'
# file6 = './results/luong_arch_32.txt'
# file7 = './results/luong_arch_31.txt'
# file8 = './results/luong_arch_21.txt'

# Performance Comparison: KukaDiverseObject
# file1 = '../results/result_ppo.txt'
# file2 = '../results/result_ipg.txt'
# file3 = '../results/ipg_her.txt'
# file4 = '../results/ipg_her_attn.txt'

# Performance comparison: KukaDiverseObject
# df1 = pd.read_csv(file1, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df1["method"] = 'ppo'
# df2 = pd.read_csv(file2, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df2["method"] = 'ipg'
# df3 = pd.read_csv(file3, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df3["method"] = 'ipg_her'
# df4 = pd.read_csv(file4, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df4["method"] = 'ipg_her_attn'

# Performance comparison: FetchReach
# df1 = pd.read_csv(file1, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df1["method"] = 'ppo'
# df2 = pd.read_csv(file2, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df2["method"] = 'ipg'
df3 = pd.read_csv(file3, sep='\t', names=['season', 'episode', 'score', 'mean_score', 'a_loss', 'c_loss'])
df3["method"] = 'ipg_her'
df3['mean_score'] = df3['mean_score'] / df3['episode']
# df4 = pd.read_csv(file4, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df4["method"] = 'ipg_her_attn'

# Attention Ablation
# df1 = pd.read_csv(file1, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df1["method"] = 'Luong Arch 1'
# df2 = pd.read_csv(file2, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df2["method"] = 'Bahdanau Arch 1'
# df3 = pd.read_csv(file3, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df3["method"] = 'Bahdanau Arch 2'
# df4 = pd.read_csv(file8, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df4["method"] = 'Luong Arch 2'
# df5 = pd.read_csv(file4, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df5["method"] = 'Bahdanau Arch 3'
# df6 = pd.read_csv(file7, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df6["method"] = 'Luong Arch 3'
# df7 = pd.read_csv(file5, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df7["method"] = 'Bahdanau Arch 4'
# df8 = pd.read_csv(file6, sep='\t', names=['season', 'score', 'mean_score', 'a_loss', 'c_loss'])
# df8["method"] = 'Luong Arch 3_2'

# join all the files by rows
#df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8]).reset_index(drop=True)
df = pd.concat([df3]).reset_index(drop=True)

print(df.head())
print(df.tail())
print(df.shape)
trunc_df = df[['season', 'a_loss', 'c_loss', 'method']]
print(trunc_df.head())
sb.set_theme()
sb.set_style('whitegrid')
g1 = sb.relplot(x='season', y='mean_score', hue='method', kind='line', data=df)
# g1._legend.set_bbox_to_anchor([0.9, 0.3])
trunc_df_melted = trunc_df.melt(id_vars=['season', 'method'], value_vars=['a_loss', 'c_loss'],\
                                var_name='loss_type', value_name='loss_value')
g2 = sb.relplot(x='season', y='loss_value', hue='loss_type', style='method', kind='line', data=trunc_df_melted)
#g2._legend.set_bbox_to_anchor([0.9, 0.8])
plt.show()


