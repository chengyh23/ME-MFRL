import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
name = 'prey'   # 'pred' or 'prey'
file_path = f'common/wandb_export_2024-05-24-{name}.csv'
df = pd.read_csv(file_path)
# filtered_df = df.loc[:, df.columns.str.contains('mfppo')]
algos = ['sac', 'mappo', 'grid_mfppo', 'me_mfppo']
# algos = ['ppo', 'sac', 'mappo', 'grid_mfppo', 'me_mfppo']
seeds = [0, 3, 7, 11, 13, 15, 18]   # , 20, 32, 42

def get_data():
    
    # basecond = np.array([[18, 20, 19, 18, 13, 4, 1],[20, 17, 12, 9, 3, 0, 0],[20, 20, 20, 12, 5, 3, 0]])
    # cond1 = np.array([[18, 19, 18, 19, 20, 15, 14],[19, 20, 18, 16, 20, 15, 9],[19, 20, 20, 20, 17, 10, 0]]) 
    # cond2 = np.array([[20, 20, 20, 20, 19, 17, 4],[20, 20, 20, 20, 20, 19, 7],[19, 20, 20, 19, 19, 15, 2]]) 
    # cond3 = np.array([[20, 20, 20, 20, 19, 17, 12],[18, 20, 19, 18, 13, 4, 1], [20, 19, 18, 17, 13, 2, 0]])    
    # return basecond, cond1, cond2, cond3
    data = []
    for algo in algos:
        tmp_df = df[[f'tag/{algo}_500x400/{seed} - R/{name}' for seed in seeds]]
        print(tmp_df.head())
        tmp = np.array(tmp_df).T
        print(tmp)
        data.append(tmp)
    return data

"""
Plot R-episode
"""
data = get_data()
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
    df[i]['algo']= algos[i] 

df=pd.concat(df) # 合并
sns.lineplot(data=df,x="episode", y="R", hue="algo", style="algo")
plt.title(f"R-{name}")
plt.savefig(f'common/R_{name}.png', dpi=300)  # You can change the filename and format as needed

# plt.show()