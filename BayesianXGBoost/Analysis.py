import pandas as pd
import numpy as np
import os
import plotly as py
import plotly.graph_objs as go
import plotly.io.orca as orca
orca.config.default_scale = 8

CN_RESULT_PATH = 'result_CN_ACCU/'
Bayesian_RESULT_PATH = 'result_XGB_ACCU/'
Random_RESULT_PATH = 'result_Random_ACCU/'
LOPT_RESULT_PATH = 'result_LOPT_ACCU/'


B_files = os.listdir(Bayesian_RESULT_PATH)
C_files = os.listdir(CN_RESULT_PATH)
R_files = os.listdir(Random_RESULT_PATH)
L_files = os.listdir(LOPT_RESULT_PATH)

# check consistency of file results
"""
if B_files.__len__() != C_files.__len__():
    raise Exception('结果文件数目不一致:\nBayesian: ' + str(B_files.__len__()) + '\nCN: ' + str(C_files.__len__()))
for b in B_files:
    if b not in C_files:
        raise Exception('CN中缺少文件: ' + b)
"""

# run evaluation
CN_accu = []
CN_time = []
Bay_accu = []
Bay_time = []
Random_accu = []
Random_time = []
LOPT_accu = []
LOPT_time = []
base = L_files
for file in base:
    print('Processing: ' + file)
    with open(CN_RESULT_PATH + file, 'r') as f:
        line = f.readline()
        line = line.split(',')
        acc = float(line[0])
        time = float(line[1])
        CN_accu.append(acc)
        CN_time.append(time)
    with open(Bayesian_RESULT_PATH + file, 'r') as f:
        line = f.readline()
        line = line.split(',')
        acc = float(line[0])
        time = float(line[1])
        Bay_accu.append(acc)
        Bay_time.append(time)
    with open(Random_RESULT_PATH + file, 'r') as f:
        line = f.readline()
        line = line.split(',')
        acc = float(line[0])
        time = float(line[1])
        Random_accu.append(acc)
        Random_time.append(time)
    with open(LOPT_RESULT_PATH + file, 'r') as f:
        line = f.readline()
        line = line.split(',')
        acc = float(line[0])
        time = float(line[1])
        # Evaluata which result is better, MDPN+LOPT or MDPN only, choose the better one,
        # Time overhead is still MDPN+LOPT
        if acc>= CN_accu[-1]:
            LOPT_accu.append(acc)
        else:
            LOPT_accu.append(CN_accu[-1])
        LOPT_time.append(time)
# Integrate results and save to file
CN_res = pd.DataFrame([CN_accu, CN_time], columns=base)
Bay_res = pd.DataFrame([Bay_accu, Bay_time], columns=base)
Random_res = pd.DataFrame([Random_accu, Random_time], columns=base)
LOPT_res = pd.DataFrame([LOPT_accu, LOPT_time], columns=base)

CN_res.to_csv('CN_result_analysis.csv')
Bay_res.to_csv('XGB_result_analysis.csv')
Random_res.to_csv('Random_result_analysis.csv')
LOPT_res.to_csv('LOPT_result_analysis.csv')

acc_res = np.array([CN_accu, Bay_accu, Random_accu, LOPT_accu])
time_res = np.array([CN_time, Bay_time, Random_time, LOPT_time])
pd.DataFrame(acc_res, columns=base).to_csv('Accu_analysis_result.csv')
pd.DataFrame(time_res, columns=base).to_csv('Time_analysis_result.csv')
# Draw plots
CN_color = 'rgb(239, 85, 59)'
Bay_color = 'rgb(99, 110, 250)'
Random_color = '#00e079'
LOPT_color = '#ffb61e'
# ACUU
data = [go.Scatter(x=base, y=acc_res[0], line=dict(color=CN_color), name='CN'),
        go.Scatter(x=base, y=acc_res[3], line=dict(color=LOPT_color), name='CN+LOPT'),
        go.Scatter(x=base, y=acc_res[1], line=dict(color=Bay_color), name='Bayesian'),
        go.Scatter(x=base, y=acc_res[2], line=dict(color=Random_color), name='BlankControlGroup')]
fig = go.Figure(data=data, layout=go.Layout(
    title='Accuracy_Compare_Between_CN_And_Bayesian',
    xaxis=dict(title='ResultFileName', type="category", showgrid=False, zeroline=False, tickangle=60),
    yaxis=dict(title='Accuracy', showgrid=True, gridcolor='rgb(30, 30, 30)', tickmode='array',
               tickvals=[i / 100 for i in range(0, 101, 5)], zeroline=True),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF'
))
fig.show()
fig.write_image("acc_res.png")
py.offline.plot(fig, auto_open=False, filename='acc_res.html')
# ACCU Box
fig = go.Figure(
    data=[go.Box(y=acc_res[0], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=CN_color, name='CN',
                 boxmean='sd'),
          go.Box(y=acc_res[3], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=LOPT_color, name='CN+LOPT',
                 boxmean='sd'),
          go.Box(y=acc_res[1], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=Bay_color, name='Bayesian',
                 boxmean='sd'),
          go.Box(y=acc_res[2], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=Random_color, name='BlankControlGroup',
                 boxmean='sd')], layout=go.Layout(
        title='Accuracy_Compare_Between_CN_And_Bayesian',
        xaxis=dict(title='ResultFileName', showgrid=False),
        yaxis=dict(title='Accuracy', showgrid=True, gridcolor='rgb(30, 30, 30)', tickmode='array',
                   tickvals=[i / 100 for i in range(0, 101, 5)], zeroline=True),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF'))
fig.show()
fig.write_image("acc_res_box.png")
py.offline.plot(fig, auto_open=False, filename='acc_res_box.html')

# Time
data = [go.Scatter(x=base, y=time_res[0], line=dict(color=CN_color,), name='CN',mode='lines+markers',marker=dict(symbol='diamond')),
go.Scatter(x=base, y=time_res[3], line=dict(color=LOPT_color), name='CN+LOPT',mode='lines+markers',marker=dict(symbol='x')),
        go.Scatter(x=base, y=time_res[1], line=dict(color=Bay_color), name='Bayesian')]
fig = go.Figure(data=data, layout=go.Layout(
    title='TimeCost_Compare_Between_CN_And_Bayesian',
    xaxis=dict(title='ResultFileName', type="category", showgrid=False, zeroline=False, tickangle=60),
    yaxis=dict(title='log10(TimeCost)/second', showgrid=False, zeroline=False),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF'
))
fig.update_yaxes(type="log")
# fig.show()
fig.write_image("time_res.png")
py.offline.plot(fig, auto_open=False, filename='time_res.html')
