# 展示 预测指标  RMSE  CE

import matplotlib.pyplot as plt
from pandas import read_excel
import matplotlib.ticker as mtick

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# data1 = read_excel('东海预测指标汇总.xlsx', sheet_name='1', header=[0, 1])
# data3 = read_excel('东海预测指标汇总.xlsx', sheet_name='3', header=[0, 1])
# data5 = read_excel('东海预测指标汇总.xlsx', sheet_name='5', header=[0, 1])
# data7 = read_excel('东海预测指标汇总.xlsx', sheet_name='7', header=[0, 1])

data1 = read_excel('台湾海峡预测指标汇总.xlsx', sheet_name='1', header=[0, 1])
data3 = read_excel('台湾海峡预测指标汇总.xlsx', sheet_name='3', header=[0, 1])
data5 = read_excel('台湾海峡预测指标汇总.xlsx', sheet_name='5', header=[0, 1])
data7 = read_excel('台湾海峡预测指标汇总.xlsx', sheet_name='7', header=[0, 1])

# ax00
data1_RMSE_MLP = data1.RMSE["MLP"]
data1_RMSE_LSTM = data1.RMSE["LSTM"]
data1_RMSE_CNN = data1.RMSE["CNN"]
data1_RMSE_CNNLSTM = data1.RMSE["CNNLSTM"]
data1_RMSE_ConvLSTMStacking = data1.RMSE["ConvLSTMStacking"]
# ax01
data1_CE_MLP = data1.CE["MLP"]
data1_CE_LSTM = data1.CE["LSTM"]
data1_CE_CNN = data1.CE["CNN"]
data1_CE_CNNLSTM = data1.CE["CNNLSTM"]
data1_CE_ConvLSTMStacking = data1.CE["ConvLSTMStacking"]
# ax10
data3_RMSE_MLP = data3.RMSE["MLP"]
data3_RMSE_LSTM = data3.RMSE["LSTM"]
data3_RMSE_CNN = data3.RMSE["CNN"]
data3_RMSE_CNNLSTM = data3.RMSE["CNNLSTM"]
data3_RMSE_ConvLSTMStacking = data3.RMSE["ConvLSTMStacking"]
# ax11
data3_CE_MLP = data3.CE["MLP"]
data3_CE_LSTM = data3.CE["LSTM"]
data3_CE_CNN = data3.CE["CNN"]
data3_CE_CNNLSTM = data3.CE["CNNLSTM"]
data3_CE_ConvLSTMStacking = data3.CE["ConvLSTMStacking"]
# ax20
data5_RMSE_MLP = data5.RMSE["MLP"]
data5_RMSE_LSTM = data5.RMSE["LSTM"]
data5_RMSE_CNN = data5.RMSE["CNN"]
data5_RMSE_CNNLSTM = data5.RMSE["CNNLSTM"]
data5_RMSE_ConvLSTMStacking = data5.RMSE["ConvLSTMStacking"]
# ax21
data5_CE_MLP = data5.CE["MLP"]
data5_CE_LSTM = data5.CE["LSTM"]
data5_CE_CNN = data5.CE["CNN"]
data5_CE_CNNLSTM = data5.CE["CNNLSTM"]
data5_CE_ConvLSTMStacking = data5.CE["ConvLSTMStacking"]
# ax30
data7_RMSE_MLP = data7.RMSE["MLP"]
data7_RMSE_LSTM = data7.RMSE["LSTM"]
data7_RMSE_CNN = data7.RMSE["CNN"]
data7_RMSE_CNNLSTM = data7.RMSE["CNNLSTM"]
data7_RMSE_ConvLSTMStacking = data7.RMSE["ConvLSTMStacking"]
# ax31
data7_CE_MLP = data7.CE["MLP"]
data7_CE_LSTM = data7.CE["LSTM"]
data7_CE_CNN = data7.CE["CNN"]
data7_CE_CNNLSTM = data7.CE["CNNLSTM"]
data7_CE_ConvLSTMStacking = data7.CE["ConvLSTMStacking"]

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 15))
# fig.tight_layout(h_pad=0.5)
labels = ['MLP', 'LSTM', 'CNN', 'CNNLSTM', 'Stacking']
colors = ['pink', 'lightblue', 'lightgreen', 'lavender', 'aqua']
# ax00
bplot00 = axs[0, 0].boxplot(x=[data1_RMSE_MLP, data1_RMSE_LSTM, data1_RMSE_CNN, data1_RMSE_CNNLSTM, data1_RMSE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[0, 0].set_title('(a)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[0, 0].set_xlabel('Models', fontsize=14)
axs[0, 0].set_ylabel('RMSE(℃)', fontsize=14)
# axs[0, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
# fill with colors
for patch, color in zip(bplot00['boxes'], colors):
    patch.set_facecolor(color)

# ax01
bplot01 = axs[0, 1].boxplot(x=[data1_CE_MLP, data1_CE_LSTM, data1_CE_CNN, data1_CE_CNNLSTM, data1_CE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[0, 1].set_title('(b)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[0, 1].set_xlabel('Models', fontsize=14)
axs[0, 1].set_ylabel('CE', fontsize=14)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'magenta', 'red']
for patch, color in zip(bplot01['boxes'], colors):
    patch.set_facecolor(color)

# ax10
bplot10 = axs[1, 0].boxplot(x=[data3_RMSE_MLP, data3_RMSE_LSTM, data3_RMSE_CNN, data3_RMSE_CNNLSTM, data3_RMSE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[1, 0].set_title('(c)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[1, 0].set_xlabel('Models', fontsize=14)
axs[1, 0].set_ylabel('RMSE(℃)', fontsize=14)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'magenta', 'red']
for patch, color in zip(bplot10['boxes'], colors):
    patch.set_facecolor(color)

# ax11
bplot11 = axs[1, 1].boxplot(x=[data3_CE_MLP, data3_CE_LSTM, data3_CE_CNN, data3_CE_CNNLSTM, data3_CE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[1, 1].set_title('(d)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[1, 1].set_xlabel('Models', fontsize=14)
axs[1, 1].set_ylabel('CE', fontsize=14)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'magenta', 'red']
for patch, color in zip(bplot11['boxes'], colors):
    patch.set_facecolor(color)

# ax20
bplot20 = axs[2, 0].boxplot(x=[data5_RMSE_MLP, data5_RMSE_LSTM, data5_RMSE_CNN, data5_RMSE_CNNLSTM, data5_RMSE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[2, 0].set_title('(e)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[2, 0].set_xlabel('Models', fontsize=14)
axs[2, 0].set_ylabel('RMSE(℃)', fontsize=14)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'magenta', 'red']
for patch, color in zip(bplot20['boxes'], colors):
    patch.set_facecolor(color)

# ax21
bplot21 = axs[2, 1].boxplot(x=[data5_CE_MLP, data5_CE_LSTM, data5_CE_CNN, data5_CE_CNNLSTM, data5_CE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[2, 1].set_title('(f)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[2, 1].set_xlabel('Models', fontsize=14)
axs[2, 1].set_ylabel('CE', fontsize=14)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'magenta', 'red']
for patch, color in zip(bplot21['boxes'], colors):
    patch.set_facecolor(color)

# ax30
bplot30 = axs[3, 0].boxplot(x=[data7_RMSE_MLP, data7_RMSE_LSTM, data7_RMSE_CNN, data7_RMSE_CNNLSTM, data7_RMSE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[3, 0].set_title('(g)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[3, 0].set_xlabel('Models', fontsize=14)
axs[3, 0].set_ylabel('RMSE(℃)', fontsize=14)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'magenta', 'red']
for patch, color in zip(bplot30['boxes'], colors):
    patch.set_facecolor(color)

# ax31
bplot31 = axs[3, 1].boxplot(x=[data7_CE_MLP, data7_CE_LSTM, data7_CE_CNN, data7_CE_CNNLSTM, data7_CE_ConvLSTMStacking],
                  vert=True,
                  patch_artist=True,
                  showmeans = False,
                  labels=labels)
axs[3, 1].set_title('(h)', y=-0.35, loc='center', weight='bold', fontsize=14)
axs[3, 1].set_xlabel('Models', fontsize=14)
axs[3, 1].set_ylabel('CE', fontsize=14)
# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen', 'magenta', 'red']
for patch, color in zip(bplot31['boxes'], colors):
    patch.set_facecolor(color)

plt.subplots_adjust(wspace=0.3, hspace=0.45)
# plt.title('title')
# plt.suptitle('East China Sea Prediction Metrics', fontsize=20, weight='bold', x=0.5, y=0.91)
plt.suptitle('Taiwan Strait Prediction Metrics', fontsize=20, weight='bold', x=0.5, y=0.91)
plt.savefig('figure2.jpg', dpi=300)
plt.show()

# print('OK')