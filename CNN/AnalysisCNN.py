import pandas as pd
import numpy as np
import os
import plotly as py
import plotly.graph_objs as go
import plotly.io.orca as orca

orca.config.default_scale = 8

CN_RESULT_PATH = 'result_CN/'
Zoopt_RESULT_PATH = 'result_zoopt/data_analysis/'
Blank_RESULT_PATH = 'result_Blank/'
LOPT_RESULT_PATH = 'result_LOPT_ACCU/'

Z_files = os.listdir(Zoopt_RESULT_PATH)
C_files = os.listdir(CN_RESULT_PATH)
B_files = os.listdir(Blank_RESULT_PATH)
# L_files = os.listdir(LOPT_RESULT_PATH)

# Check file consistency
"""
if B_files.__len__() != C_files.__len__():
    raise Exception('结果文件数目不一致:\nZoopt: ' + str(B_files.__len__()) + '\nCN: ' + str(C_files.__len__()))
for b in B_files:
    if b not in C_files:
        raise Exception('CN中缺少文件: ' + b)
"""

# Run Evaluation
CN_accu = []
CN_time = []
Zoopt_accu = []
Zoopt_time = []
Blank_accu = []
Blank_time = []
LOPT_accu = []
LOPT_time = []
base = C_files
for file in base:
    if base != Z_files:
        print('Process files: ' + file)
        if 'mnist' in file:
            num = int(file[12:-8])
            filename = 'mnist_' + str(num) + '.txt'
        else:
            num = int(file[11:-8])
            filename = 'svhn_' + str(num) + '.txt'

        with open(CN_RESULT_PATH + file, 'r') as f:
            line = f.readline()
            line = line.split(',')
            acc = float(line[0])
            time = float(line[1])
            CN_accu.append(acc)
            CN_time.append(time)
        with open(Zoopt_RESULT_PATH + filename, 'r') as f:
            line = f.readline()
            line = line.split(',')
            acc = float(line[0])
            time = float(line[1])
            Zoopt_accu.append(acc)
            Zoopt_time.append(time)
    else:
        print('Process files: ' + file)
        if 'mnist' in file:
            num = int(file[6:-4])
            filename = 'mnist_' + str(num) + '.pkl.txt'
        else:
            num = int(file[5:-4])
            filename = 'svhn_subset' + str(num) + '.pkl.txt'

        with open(CN_RESULT_PATH + filename, 'r') as f:
            line = f.readline()
            line = line.split(',')
            acc = float(line[0])
            time = float(line[1])
            CN_accu.append(acc)
            CN_time.append(time)
        with open(Zoopt_RESULT_PATH + file, 'r') as f:
            line = f.readline()
            line = line.split(',')
            acc = float(line[0])
            time = float(line[1])
            Zoopt_accu.append(acc)
            Zoopt_time.append(time)
    with open(Blank_RESULT_PATH + file, 'r') as f:
        line = f.readline()
        line = line.split(',')
        acc = float(line[0])
        time = float(line[1])
        Blank_accu.append(acc)
        Blank_time.append(time)
    # with open(LOPT_RESULT_PATH + file, 'r') as f:
    #     line = f.readline()
    #     line = line.split(',')
    #     acc = float(line[0])
    #     time = float(line[1])
    #     # 这里判断一下对于同一个文件，CN+LOPT和LOPT优化后谁的结果好，取好的那个，时间仍然按照CN+LOPT计算
    #     if acc>= CN_accu[-1]:
    #         LOPT_accu.append(acc)
    #     else:
    #         LOPT_accu.append(CN_accu[-1])
    #     LOPT_time.append(time)
# integrate results and save to file
CN_res = pd.DataFrame([CN_accu, CN_time], columns=base)
Zoopt_res = pd.DataFrame([Zoopt_accu, Zoopt_time], columns=base)
Blank_res = pd.DataFrame([Blank_accu, Blank_time], columns=base)
# LOPT_res = pd.DataFrame([LOPT_accu, LOPT_time], columns=base)

CN_res.to_csv('CN_result_analysis.csv')
Zoopt_res.to_csv('Zoopt_result_analysis.csv')
Blank_res.to_csv('Blank_result_analysis.csv')
# LOPT_res.to_csv('LOPT_result_analysis.csv')

# acc_res = np.array([CN_accu, Zoopt_accu, Blank_accu, LOPT_accu])
# time_res = np.array([CN_time, Zoopt_time, Blank_time, LOPT_time])
acc_res = np.array([CN_accu, Zoopt_accu, Blank_accu])
time_res = np.array([CN_accu, Zoopt_time, Blank_time])
pd.DataFrame(acc_res, columns=base).to_csv('Accu_analysis_result.csv')
pd.DataFrame(time_res, columns=base).to_csv('Time_analysis_result.csv')
# Draw figures
CN_color = 'rgb(239, 85, 59)'
Zoopt_color = 'rgb(99, 110, 250)'
Blank_color = '#00e079'
LOPT_color = '#ffb61e'
x = []
for name in base:
    x.append((name.replace('subset', '')).replace('.pkl.txt', ''))
# ACUU
# data = [go.Scatter(x=x, y=acc_res[0], line=dict(color=CN_color), name='CN'),
#         go.Scatter(x=x, y=acc_res[3], line=dict(color=LOPT_color), name='CN+LOPT'),
#         go.Scatter(x=x, y=acc_res[1], line=dict(color=Bay_color), name='Zoopt'),
#         go.Scatter(x=x, y=acc_res[2], line=dict(color=Blank_color), name='BlankControlGroup')]
data = [go.Scatter(x=x, y=acc_res[0], line=dict(color=CN_color), name='CN',mode='lines+markers',marker=dict(symbol='diamond')),
        go.Scatter(x=x, y=acc_res[1], line=dict(color=Zoopt_color), name='Zoopt',mode='lines+markers',marker=dict(symbol='x')),
        go.Scatter(x=x, y=acc_res[2], line=dict(color=Blank_color), name='BCG')]
fig = go.Figure(data=data, layout=go.Layout(
    xaxis=dict(title='Sample', type="category", showgrid=False, zeroline=False, tickangle=60,showticklabels=False),
    yaxis=dict(title='Accuracy', showgrid=True, gridcolor='rgb(30, 30, 30)', tickmode='array',
               tickvals=[i / 100 for i in range(0, 101, 5)], zeroline=True),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF'
))
fig.show()
# fig.write_image("acc_res.png")
py.offline.plot(fig, auto_open=False, filename='acc_res.html')
# ACCU Box
# fig = go.Figure(
#     data=[go.Box(y=acc_res[0], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=CN_color, name='CN',
#                  boxmean='sd'),
#           go.Box(y=acc_res[3], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=LOPT_color, name='CN+LOPT',
#                  boxmean='sd'),
#           go.Box(y=acc_res[1], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=Zoopt_color, name='Zoopt',
#                  boxmean='sd'),
#           go.Box(y=acc_res[2], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=Blank_color, name='BlankControlGroup',
#                  boxmean='sd')], layout=go.Layout(
#         title='Accuracy Compare',
#         xaxis=dict(title='Sample Name', showgrid=False),
#         yaxis=dict(title='Accuracy', showgrid=True, gridcolor='rgb(30, 30, 30)', tickmode='array',
#                    tickvals=[i / 100 for i in range(0, 101, 5)], zeroline=True),
#         paper_bgcolor='#FFFFFF',
#         plot_bgcolor='#FFFFFF'))
fig = go.Figure(
    data=[go.Box(y=acc_res[0], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=CN_color, name='CN',
                 boxmean='sd'),
          go.Box(y=acc_res[1], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=Zoopt_color, name='Zoopt',
                 boxmean='sd'),
          go.Box(y=acc_res[2], boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=Blank_color,
                 name='BCG', boxmean='sd')], layout=go.Layout(
        xaxis=dict(title='Sample', showgrid=False),
        yaxis=dict(title='Accuracy', showgrid=True, gridcolor='rgb(30, 30, 30)', tickmode='array',
                   tickvals=[i / 100 for i in range(0, 101, 5)], zeroline=True),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF'))
fig.show()
# fig.write_image("acc_res_box.png")
py.offline.plot(fig, auto_open=False, filename='acc_res_box.html')

# Time
# data = [go.Scatter(x=x, y=time_res[0], line=dict(color=CN_color,), name='CN',mode='lines+markers',marker=dict(symbol='diamond')),
#         go.Scatter(x=x, y=time_res[3], line=dict(color=LOPT_color), name='CN+LOPT',mode='lines+markers',marker=dict(symbol='x')),
#         go.Scatter(x=x, y=time_res[1], line=dict(color=Zoopt_color), name='Zoopt')]
data = [go.Scatter(x=x, y=time_res[0], line=dict(color=CN_color, ), name='CN', mode='lines+markers',
                   marker=dict(symbol='diamond')),
        go.Scatter(x=x, y=time_res[1], line=dict(color=Zoopt_color), name='Zoopt')]

fig = go.Figure(data=data, layout=go.Layout(
    xaxis=dict(title='Sample', type="category", showgrid=False, zeroline=False, tickangle=60,showticklabels=False),
    yaxis=dict(title='log10(TimeCost)/second', showgrid=False, zeroline=False),
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FFFFFF'
))
fig.update_yaxes(type="log")
fig.show()
# fig.write_image("time_res.png")
py.offline.plot(fig, auto_open=False, filename='time_res.html')
