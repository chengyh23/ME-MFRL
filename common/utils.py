"""
Plot R-episode
强化学习,seaborn
"""
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_data(file_path):
    df = pd.read_csv(file_path)
    # filtered_df = df.loc[:, df.columns.str.contains('mfppo')]
    algos = ['sac', 'mappo', 'grid_mfppo', 'me_mfppo']
    # algos = ['ppo', 'sac', 'mappo', 'grid_mfppo', 'me_mfppo']
    seeds = [0, 3, 7, 11, 13, 15, 18]   # , 20, 32, 42
    data = []
    for algo in algos:
        tmp_df = df[[f'tag/{algo}_500x400/{seed} - R/{name}' for seed in seeds]]
        print(tmp_df.head())
        tmp = np.array(tmp_df).T
        print(tmp)
        data.append(tmp)
    return data

    name = 'prey'   # 'pred' or 'prey'
    # file_path = f'common/wandb_export_2024-05-24-{name}.csv'
    file_path = f'common/wandb_export_2024-06-06-MvN-{name}.csv'
    conf = file_path.strip('.csv')
    data = get_data2(file_path)
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
        df[i]['algo']= algos[i] 

    df=pd.concat(df) # 合并
    sns.lineplot(data=df,x="episode", y="R", hue="algo", style="algo")
    plt.title(f"R-{conf}")
    plt.savefig(f'common/R-{conf}.png', dpi=300)
    # plt.savefig(f'common/R_{name}.png', dpi=300)  # You can change the filename and format as needed

    # plt.show()
    
def get_data2():
    for name in ['pred', 'prey']:
        # file_path = f'common/wandb_export_2024-05-24-{name}.csv'
        file_path = f'common/wandb_export_2024-06-06-MvN-{name}.csv'
        conf = file_path.strip('.csv')
        
        df = pd.read_csv(file_path)
        seeds = [0, 3]   # , 20, 32, 42
        data = []
        MvNs = ['30v10', '20v20', '10v30']
        for MvN in MvNs:
            tmp_df = df[[f'myr1/dqn_500x400/{MvN}/{seed} - R/{name}' for seed in seeds]]
            print(tmp_df.head())
            tmp = np.array(tmp_df).T
            data.append(tmp)
        # return data

    
        # data = get_data2(file_path)
        df=[]
        for i in range(len(data)):
            df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
            df[i]['MvN']= MvNs[i]

        df=pd.concat(df) # 合并
        sns.lineplot(data=df,x="episode", y="R", hue="MvN", style="MvN")
        plt.title(f"R-{name}")
        plt.savefig(f'common/MvN-R_{name}.png', dpi=300)
        # plt.savefig(f'common/R_{name}.png', dpi=300)  # You can change the filename and format as needed

        # plt.show()
        plt.clf()
    
if __name__ == '__main__':
    get_data2()