import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import plotly
# import plotly_express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff  
from plotly.subplots import make_subplots  

outfile_root="./results2/"

# from utils import loadlog, checklogvalid
strategies_order = ["Janosov", "DRL", "PUARL"]
strategies_abbrv2full = {
    "SC": "Surge-Cast", 
    "BF_2": "Infotaxis", #!
    "IT_2": "Infotaxis", 
    "BF_RELDre": "GasHunter"}

strategies_colors_dict = {
    "Janosov": '#31797c', 
    # "Best-First": '#035a8c', 
    "DRL": '#1c475d', 
    "PUARL": '#ac536d'}
strategies_patterns_dict = {
    "Janosov": ".", 
    # "Best-First": "x", 
    "DRL": "+",
    "PUARL": '/'}    #,'-', '\\'
strategies_markersymbol_dict = {
    "Janosov": 4,
    # "Best-First": 17,
    "DRL": 2,
    "PUARL": 3}

metrics = ["success_rate", "avg_steps"]
metrics_names_dict = {"success_rate": "Success Rate", "avg_steps": "Avg Steps"}
metrics_colors_dict = {"success_rate": "#ac536d", "avg_steps": "#035a8c", "Etraj": "#31797c"}
metrics_patterns_dict = {"success_rate": ".", "avg_steps": "x", "Etraj": "+"}
metrics_markersymbol_dict = {"success_rate": 4, "avg_steps": 17, "Etraj": 2}

# Common settings

linewidth = 6
markersize = 30
fontsize = 30
titlefontsize = 30

def update_fig(fig, xtitle, ytitle, ycolor=None, yaxisrange=None, ytickformat='.2f', 
               ytitle2=None, ycolor2=None, yaxisrange2=None, ytickformat2="",
               ytitle3=None, ycolor3=None, yaxisrange3=None, xaxisdomain=None, ytickformat3="",
               margin={'l': 15, 'r': 15, 't': 0, 'b': 0}, width = 800, height = 450, 
               xtickformat='.2f', 
               xcategoryorder='trace', xcategoryarray=None,
               ):
    fig.update_layout(
        xaxis_title=xtitle, xaxis_title_font=dict(size=titlefontsize),
        yaxis_title=ytitle, yaxis_title_font=dict(size=titlefontsize),
        font=dict(size=fontsize, family='Arial', color='rgb(0, 0, 0)'),
        margin=margin,
        width=width, height=height,
        legend=dict(
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            orientation="h",
        ),
        legend_title="",
        plot_bgcolor='rgb(255,255,255)',
        xaxis=dict(
            showline=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            ticks='outside',
            tickformat=xtickformat,
            gridcolor='rgb(245, 245, 245)',
            mirror=True,
            categoryorder=xcategoryorder, categoryarray=xcategoryarray,
            domain=xaxisdomain,
        ),
        yaxis=dict(
            color=ycolor,
            showline=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            ticks='outside',
            tickformat=ytickformat,
            gridcolor='rgb(245, 245, 245)',
            mirror=True,
            range=yaxisrange,
        ),
    )
    if ytitle2 is not None:
        """ https://stackoverflow.com/questions/29046057/plotly-grouped-bar-chart-with-multiple-axes """
        fig.update_layout(
            yaxis2=dict(
                color=ycolor2,
                title=ytitle2,
                overlaying='y',
                side='right',
                ticks='outside',
                tickformat=ytickformat2,
                range=yaxisrange2,
                anchor='x',
            ),
            boxmode='group',
        )
    if ytitle3 is not None:
        fig.update_layout(
            yaxis3=dict(
                color=ycolor3,
                title=ytitle3,
                overlaying='y',
                side='right',
                ticks='outside',
                tickformat=ytickformat3,
                range=yaxisrange3,
                anchor='free',
                position=1.0,
            ),
            boxmode='group',
        )
    return fig

# plot using csv
import ast

outfile_root="common/fig/"
print(outfile_root)

# TODO assert uniqueness of index in the csv file (.iloc[0])
def N_():
    _names = [
        'se-R10/no1/janosov_100x100/3v1/1',
        # 'se-R10/no1/janosov_100x100/4v1/1',
        'se-R10/no1/janosov_100x100/5v1/1',
        # 'se-R10/no1/janosov_100x100/6v1/1',
        'se-R10/no1/janosov_100x100/7v1/1',
        
        're-Rsrspure/no/eg/dqn_Xx400/3v1/0',
        # 're-Rsrspure/no/eg_2/dqn_Xx400/4v1/0',
        're-Rsrspure/no/eg/dqn_Xx400/5v1/0',
        # 're-Rsrspure/no/eg_2/dqn_Xx400/6v1/0',
        're-Rsrspure/no/eg/dqn_Xx400/7v1/0',
        
        're-Rsrspure/no/ka_cv/dqn_Xx400/3v1/0',
        # 're-Rsrspure/no/ka_cv_2/dqn_Xx400/4v1/0',
        're-Rsrspure/no/ka_cv/dqn_Xx400/5v1/0',
        # 're-Rsrspure/no/ka_cv_2/dqn_Xx400/6v1/0',
        're-Rsrspure/no/ka_cv/dqn_Xx400/7v1/0',
    ]
    
    data = {
        'algo': ["Janosov"] * 3 + ["DRL"] * 3 + ["PUARL"] * 3,
        'N': [3, 5, 7] * 3,
        # 'algo': ["Janosov"] * 5 + ["DRL"] * 5 + ["PUARL"] * 5,
        # 'N': [3, 4, 5, 6, 7] * 3,
        'success_rate': None,
        'avg_steps': None,
    }
    _df_in = pd.read_csv('common/csv/test_log.csv', index_col=False)
    print(_df_in.head())
    _sr = []
    _steps = []
    for _name in _names:
        idx = _df_in['config']==_name
        _sr.append(_df_in[idx]['success_rate'].iloc[0])
        _steps.append(_df_in[idx]['avg_steps'].iloc[0])
    data['success_rate'] = _sr
    data['avg_steps'] = _steps
    
    df = pd.DataFrame(data)
    print(df)

    for metric in metrics:  # "success_rate", ""
        
        fig = go.Figure()
        # fig.add_trace(go.Bar(x=gpb["robots_tau"], y=gpb["success_rate"],
        #                          yaxis='y', offsetgroup=1,
        #                          marker=dict(line_color=metrics_colors_dict["success_rate"], color=metrics_colors_dict["success_rate"], 
        #                                         pattern_shape=metrics_patterns_dict["success_rate"]),
        #                         width= 0.2, ))
        # fig.update_traces(
        #     marker=dict(pattern_fillmode="replace", line_width=6.0, pattern_size=7),
        # )
        for algo in strategies_order:
            gpb = df[df['algo'] == algo]

            print(gpb)
            # fig.add_trace(go.Scatter(x=gpb["N"], y=gpb[metric],
            #                         yaxis='y', 
            #                         marker=dict(size=markersize, symbol=metrics_markersymbol_dict[metric], color=strategies_colors_dict[algo]),
            #                         line=dict(color=strategies_colors_dict[algo], width=linewidth),
            #                         showlegend=True, name=algo
            # ))
            fig.add_trace(go.Bar(x=gpb["N"], y=gpb[metric],
                                    yaxis='y', # offsetgroup=1,
                                    marker=dict(line_color=strategies_colors_dict[algo], color=strategies_colors_dict[algo], 
                                                    pattern_shape=strategies_patterns_dict[algo]),
                                    width= 0.4,
                                    showlegend=True, name=algo
            ))
        fig = update_fig(fig, 'N', metrics_names_dict[metric], xtickformat='.d', ytickformat='.1f',
                        width=750, height=450,
                        margin={'l': 15, 'r': 15, 't': 15, 'b': 0})
        # fig.show()
        # filename = outfile_root+f'tau_{N}robot_successrate.png'
        # fig.write_image(filename, scale=1)
        # print(filename)

        # # SEARCH TIME + PATH EFFICIENCY
        # df = df[df['success'] == True]  #! Search time & Path Efficiency For success arrival only
        # # fig = go.Figure()
        # # fig.add_trace(go.Box(x=df['robots_tau'], y=df['searchtime'], 
        # #                     yaxis='y', offsetgroup=1,
        # #                     line=dict(color=metrics_colors_dict['searchtime'], )))
        # fig.add_trace(go.Box(x=df['robots_tau'], y=df['Etraj'],
        #                     yaxis='y2', offsetgroup=2,
        #                     line=dict(color=metrics_colors_dict['Etraj']),
        #                     width=0.2,
        #                     showlegend=False))
        # fig = update_fig(fig, 'log(tau)', 'Success Rate', ycolor=metrics_colors_dict['success_rate'], ytickformat='.1f',
        #                  ytitle2='Path Efficiency', ycolor2=metrics_colors_dict['Etraj'],
        #                  width=750, height=450,
        #                  margin={'l': 15, 'r': 15, 't': 15, 'b': 0})

        # fig.show()
        filename = outfile_root+f'N_{metric}.png'
        print(filename)
        fig.write_image(filename, scale=1)
    
def NF_():
    _names = [
        "se-R10/no1/janosov_100x100/3v1/1",
        "se-R10/no1/janosov_100x100/3v1/1",
        "se-R10/no1/janosov_100x100/3v1/1",
        "re-Rsrspure/no/eg/dqn_Xx400/3v1/0",
        "re-Rsrspure/no/eg/dqn_Xx400/3v1/0",
        "re-Rsrspure/no/eg/dqn_Xx400/3v1/0",
        "re-Rsrspure/no/ka_cv/dqn_Xx400/3v1/0",
        "re-Rsrspure/no/ka_cv/dqn_Xx400/3v1/0",
        "re-Rsrspure/no/ka_cv/dqn_Xx400/3v1/0",
    ]
    
    _cmds = [
        "baseline/janosov.py --algo janosov --test --test_n_round 100 --num_adversaries 3 --noisy_obs",
        "baseline/janosov.py --algo janosov --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor 2",
        "baseline/janosov.py --algo janosov --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor 4",    
        "test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --idx 4999",
        "test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor 2 --idx 4999",
        "test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor 4 --idx 4999",
        "test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --use_kf_act --kf_proc_model cv --idx 4999",
        "test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor 2 --use_kf_act --kf_proc_model cv --idx 4999",
        "test_my.py --algo dqn --test --test_n_round 100 --num_adversaries 3 --noisy_obs --test_noisy_factor 4 --use_kf_act --kf_proc_model cv --idx 4999",
    ]

    data = {
        'algo': ["Janosov"] * 3 + ["DRL"] * 3 + ["PUARL"] * 3,
        'NF': [1, 2, 4] * 3,
        'success_rate': None,
        'avg_steps': None,
    }
    _df_in = pd.read_csv('common/csv/test_log_NF.csv', index_col=False)
    print(_df_in.head())
    _sr = []
    _steps = []
    for i in range(len(_names)):
        _name = _names[i]
        _cmd = _cmds[i]
        # idx = _df_in['config']==_name and _df_in['cmd']==_cmd
        idx = _df_in['cmd']==_cmd
        _sr.append(_df_in[idx]['success_rate'].iloc[0])
        _steps.append(_df_in[idx]['avg_steps'].iloc[0])
    data['success_rate'] = _sr
    data['avg_steps'] = _steps
    
    df = pd.DataFrame(data)
    df['NF'] = df['NF'].astype(str)
    print(df)
    for metric in metrics:
        fig = go.Figure()
        for algo in strategies_order:
            gpb = df[df['algo'] == algo]

            print(gpb)
            # fig.add_trace(go.Scatter(x=gpb["NF"], y=gpb[metric],
            #                         yaxis='y', 
            #                         marker=dict(size=markersize, symbol=metrics_markersymbol_dict[metric], color=strategies_colors_dict[algo]),
            #                         line=dict(color=strategies_colors_dict[algo], width=linewidth),
            #                         showlegend=True, name=algo
            # ))
            fig.add_trace(go.Bar(x=gpb["NF"], y=gpb[metric],
                                    yaxis='y', # offsetgroup=1,
                                    marker=dict(line_color=strategies_colors_dict[algo], color=strategies_colors_dict[algo], 
                                                    pattern_shape=strategies_patterns_dict[algo]),
                                    width= 0.2,
                                    showlegend=True, name=algo
            ))
        fig = update_fig(fig, 'noise factor', metrics_names_dict[metric], ytickformat='.1f',
                        width=750, height=450,
                        margin={'l': 15, 'r': 15, 't': 15, 'b': 0})
        
        filename = outfile_root+f'NF_{metric}.png'
        print(filename)
        fig.write_image(filename, scale=1)
    
def eps_k_():
    _names = [
        're-Rsrspure/no/eg/dqn_Xx400/3v1/0',
        're-Rsrspure/no/eg_4/dqn_Xx400/3v1/0',
        're-Rsrspure/no/eg_6/dqn_Xx400/3v1/0',
        're-Rsrspure/no/ka_cv/dqn_Xx400/3v1/0',
        're-Rsrspure/no/ka_cv_4/dqn_Xx400/3v1/0',
        're-Rsrspure/no/ka_cv_6/dqn_Xx400/3v1/0',
    ]
    data = {
        'algo': ["DRL", "DRL", "DRL", "PUARL", "PUARL", "PUARL"],
        'eps_k': [0.2, 0.4, 0.6] * 2,
        # 'success_rate': [0.56, 0.69, 0.8, 0.76, 0.77, 0.9, 0.81, 0.89, 0.98],
        'success_rate': None,
        'avg_steps': None,
    }
    _df_in = pd.read_csv('common/csv/test_log.csv', index_col=False)
    print(_df_in.head())
    _sr = []
    _steps = []
    for _name in _names:
        idx = _df_in['config']==_name
        _sr.append(_df_in[idx]['success_rate'].iloc[0])
        _steps.append(_df_in[idx]['avg_steps'].iloc[0])
    # _df_in = _df_in[_df_in['config'].isin(_names)]
    # data['success_rate'] = list(_df_in['success_rate'])
    # data['avg_steps'] = list(_df_in['avg_steps'])
    data['success_rate'] = _sr
    data['avg_steps'] = _steps
    
    df = pd.DataFrame(data)
    # df['eps_k'] = df['eps_k'].astype(str)
    print(df)
    for metric in metrics:  # "success_rate", 
        fig = go.Figure()
        # for algo in strategies_order:
        for algo in ["DRL", "PUARL",]:
            gpb = df[df['algo'] == algo]

            print(gpb)
            # fig.add_trace(go.Scatter(x=gpb["eps_k"], y=gpb[metric],
            #                         yaxis='y', 
            #                         marker=dict(size=markersize, symbol=metrics_markersymbol_dict[metric], color=strategies_colors_dict[algo]),
            #                         line=dict(color=strategies_colors_dict[algo], width=linewidth),
            #                         showlegend=True, name=algo
            # ))
            fig.add_trace(go.Bar(x=gpb["eps_k"], y=gpb[metric],
                                    yaxis='y', # offsetgroup=1,
                                    marker=dict(line_color=strategies_colors_dict[algo], color=strategies_colors_dict[algo], 
                                                    pattern_shape=strategies_patterns_dict[algo]),
                                    width= 0.06,
                                    showlegend=True, name=algo
            ))
        fig = update_fig(fig, 'h', metrics_names_dict[metric], ytickformat='.1f',
                        width=750, height=450,
                        margin={'l': 15, 'r': 15, 't': 15, 'b': 0})
        
        filename = outfile_root+f'eps_{metric}.png'
        print(filename)
        fig.write_image(filename, scale=1) 
if __name__ == "__main__":
    N_()
    # NF_()
    # eps_k_()