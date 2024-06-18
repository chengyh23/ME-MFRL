"""
Plot R-episode
强化学习,seaborn
"""
import os
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

def smooth_data(data, window_size):
    # Calculate the moving average for smoothing
    return data.rolling(window=window_size).mean()

def get_data3(seeds=[0, 3, 7], smoothing_tensorboard=0.8):
    file_path = f'common/wandb_export_2024-06-12T09_16_23.052+08_00.csv'
    conf = file_path.strip('.csv')
    df = pd.read_csv(file_path)
    
    data = []
    groups = ['ao', 'no']
    group_name = 'noisy'
    for noisy in groups:
        tmp_df = df[[f'se-R10/{noisy}/eg/dqn_500x400/3v1/{seed} - R/pred' for seed in seeds]]
        print(tmp_df.head())
        tmp_df_smooth = tmp_df.ewm(alpha=1-smoothing_tensorboard).mean()
        print(tmp_df_smooth.head())
        
        tmp = np.array(tmp_df_smooth).T
        data.append(tmp)

    # return data
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
        df[i][group_name]= groups[i]
    df=pd.concat(df) # 合并
    print(df.head())
    sns.lineplot(data=df,x="episode", y="R", hue=group_name, style=group_name)
    plt.title(f"{group_name}-R_pred")
    plt.savefig(f'common/{group_name}-R_pred.png', dpi=300)
    plt.clf()
def get_data4(seeds=[0, 3, 7], smoothing_tensorboard=0.99):
    file_path = f'common/wandb_export_2024-06-12-no.csv'
    conf = file_path.strip('.csv')
    df = pd.read_csv(file_path)
    
    data = []
    groups = ['eg', 'ka_rw', 'ka_cv']
    group_name = 'act'
    for act in groups:
        tmp_df = df[[f'se-R10/no/{act}/dqn_500x400/3v1/{seed} - R/pred' for seed in seeds]]
        print(tmp_df.head())
        tmp_df_smooth = tmp_df.ewm(alpha=1-smoothing_tensorboard).mean()
        # print(tmp_df_smooth.head())
        tmp = np.array(tmp_df_smooth).T
        data.append(tmp)

    # return data
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
        df[i][group_name]= groups[i]
    df=pd.concat(df) # 合并
    print(df.head())
    sns.lineplot(data=df,x="episode", y="R", hue=group_name, style=group_name)
    plt.title(f"{group_name}-R_pred")
    plt.savefig(f'common/{group_name}-R_pred.png', dpi=300)
    plt.clf()
    
def get_data5(seeds=[0, 3, 7], smoothing_tensorboard=0.99, y_range=(-1.5,0.25)):
    """ Impact of noisy_factor """
    out_dir = 'common/fig'
    file_dir = 'common/csv'
    file_paths = ['wandb_export_2024-06-12-no.csv', \
                  'wandb_export_2024-06-13-no2.csv', 'wandb_export_2024-06-13-no10.csv']
    
    data = []
    act = 'eg'
    # groups = ['eg', 'ka_rw', 'ka_cv']
    groups = ['no', 'no2', 'no10']
    group_name = 'noisy_factor'
    for i,group in enumerate(groups):
        conf = file_paths[i].strip('.csv')
        df = pd.read_csv(os.path.join(file_dir, file_paths[i]))
        tmp_df = df[[f'se-R10/{group}/{act}/dqn_500x400/3v1/{seed} - R/pred' for seed in seeds]]
        print(tmp_df.head())
        tmp_df_smooth = tmp_df.ewm(alpha=1-smoothing_tensorboard).mean()
        # print(tmp_df_smooth.head())
        tmp = np.array(tmp_df_smooth).T
        data.append(tmp)

    # return data
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
        df[i][group_name]= groups[i]
    df=pd.concat(df) # 合并
    print(df.head())
    sns.lineplot(data=df,x="episode", y="R", hue=group_name, style=group_name)
    if y_range is not None:
        plt.ylim(y_range)
    plt.title(f"{group_name}-R_pred-{act}")
    plt.savefig(os.path.join(out_dir, f'{group_name}-R_pred-{act}.png'), dpi=300)
    plt.clf()  

def get_data6(seeds=[0, 3, 7], smoothing_tensorboard=0.99, y_range=(-1.5,0.25)):
    """ Impact of # pursuers """
    out_dir = 'common/fig'
    file_dir = 'common/csv'
    file_paths = ['wandb_export_2024-06-12-3v1.csv', \
                  'wandb_export_2024-06-13-5v1.csv', 'wandb_export_2024-06-13-7v1.csv']
    
    data = []
    act = 'ka_cv'
    # groups = ['eg', 'ka_rw', 'ka_cv']
    groups = ['3v1', '5v1', '7v1']
    group_name = 'num_pursuer'
    for i,group in enumerate(groups):
        conf = file_paths[i].strip('.csv')
        df = pd.read_csv(os.path.join(file_dir, file_paths[i]))
        _noisy = 'no' if group=='3v1' else 'no1'
        tmp_df = df[[f'se-R10/{_noisy}/{act}/dqn_500x400/{group}/{seed} - R/pred' for seed in seeds]]
        print(tmp_df.head())
        tmp_df_smooth = tmp_df.ewm(alpha=1-smoothing_tensorboard).mean()
        # print(tmp_df_smooth.head())
        tmp = np.array(tmp_df_smooth).T
        data.append(tmp)

    # return data
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
        df[i][group_name]= groups[i]
    df=pd.concat(df) # 合并
    print(df.head())
    sns.lineplot(data=df,x="episode", y="R", hue=group_name, style=group_name)
    if y_range is not None:
        plt.ylim(y_range)
    plt.title(f"{group_name}-R_pred-{act}")
    plt.savefig(os.path.join(out_dir, f'{group_name}-R_pred-{act}.png'), dpi=300)
    plt.clf()  

def get_data7(seeds=[0, 3, 7], smoothing_tensorboard=0.99, y_range=(-1.5,0.25)):
    """ Impact of Eps """
    out_dir = 'common/fig'
    file_dir = 'common/csv'
    file_paths = ['wandb_export_2024-06-12-3v1.csv', \
                  'wandb_export_2024-06-13-E4.csv', 'wandb_export_2024-06-13-E6.csv']
    
    data = []
    act = 'ka_rw'
    # groups = ['eg', 'ka_rw', 'ka_cv']
    groups = ['3v1', 'E4', 'E6']
    group_name = 'Eps'
    for i,group in enumerate(groups):
        conf = file_paths[i].strip('.csv')
        df = pd.read_csv(os.path.join(file_dir, file_paths[i]))
        _eps = '' if group=='3v1' else f'-{group}'
        _noisy = 'no' if group=='3v1' else 'no1'
        tmp_df = df[[f'se-R10{_eps}/{_noisy}/{act}/dqn_500x400/3v1/{seed} - R/pred' for seed in seeds]]
        print(tmp_df.head())
        tmp_df_smooth = tmp_df.ewm(alpha=1-smoothing_tensorboard).mean()
        # print(tmp_df_smooth.head())
        tmp = np.array(tmp_df_smooth).T
        data.append(tmp)

    # return data
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='R'))
        df[i][group_name]= groups[i]
    df=pd.concat(df) # 合并
    print(df.head())
    sns.lineplot(data=df,x="episode", y="R", hue=group_name, style=group_name)
    if y_range is not None:
        plt.ylim(y_range)
    plt.title(f"{group_name}-R_pred-{act}")
    plt.savefig(os.path.join(out_dir, f'{group_name}-R_pred-{act}.png'), dpi=300)
    plt.clf()  
    
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

def get_tfboard_data(metric='Agent_0_kill_op', seeds=[0, 3, 7], smoothing_tensorboard=0.99):
    group_name = 'obs_act'
    groups = ['ao-eg', 'no-eg', 'no-ka_rw', 'no-ka_cv']
    
    data = []
    for group in groups:
        repeats = []
        for seed in seeds:
            # in_path = 'data/tmp/se-R10-ao-eg-dqn_500x400-3v1-0/dqn/events.out.tfevents.1718095850.inspur-NF5468M5'
            in_path = f'data/tmp/se-R10-{group}-dqn_500x400-3v1-{seed}/dqn' # TODO what if multiple tfevents?
            
            event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
            event_data.Reload()  # synchronously loads all of the data written so far b
            # print(event_data.Tags())  # print all tags
            keys = event_data.scalars.Keys()  # get all tags,save in a list
            print(keys)
            df = pd.DataFrame(columns=keys)
            for key in tqdm(keys):
                df[key] = pd.DataFrame(event_data.Scalars(key)).value
            # print(df.head())
            tmp_df_smooth = df.ewm(alpha=1-smoothing_tensorboard).mean()
            tmp = np.array(tmp_df_smooth[metric])
            repeats.append(tmp)
        repeats = np.vstack(repeats)
        data.append(repeats)
    # return data
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name=metric))
        df[i][group_name]= groups[i]
    df=pd.concat(df) # 合并
    print(df.head())
    sns.lineplot(data=df,x="episode", y=metric, hue=group_name, style=group_name)
    plt.title(f"{group_name}-{metric}")
    plt.savefig(f'common/{group_name}-{metric}.png', dpi=300)
    plt.clf()
# import tensorflow as tf
# def read_event_file():
#     event_file = 'data/tmp/se-R10-no-eg-dqn_500x400-3v1-0/dqn/events.out.tfevents.1718094091.inspur-NF5468M5'
#     event_file = 'data/tmp/se-R10-no-eg-dqn_500x400-3v1-0/dqn/events.out.tfevents.1718096031.inspur-NF5468M5'
#     events = []
#     for event in tf.compat.v1.train.summary_iterator(event_file):
#         for value in event.summary.value:
#             events.append({
#                 'step': event.step,
#                 'wall_time': event.wall_time,
#                 'tag': value.tag,
#                 'value': value.simple_value
#             })
#     print(events)
#     # return events

# Example usage


if __name__ == '__main__':
    get_data7()
    # get_tfboard_data(metric='Agent_0_kill_op')
    # get_tfboard_data(metric='Agent_0_step_ct_op')
    # read_event_file()