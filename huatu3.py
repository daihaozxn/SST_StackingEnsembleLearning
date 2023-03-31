## 对于不同 lead time，将台湾海峡和东海 个体模型和元模型的 最优预测指标 展示出来

import matplotlib.pyplot as plt
from pandas import read_excel
from matplotlib import ticker as mtick
from matplotlib.pyplot import MultipleLocator

data_TaiwanStrait = read_excel('台湾海峡和东海最优预测指标汇总.xlsx', sheet_name='TaiwanStrait', header=[0, 1], index_col=0)
data_EastChinaSea = read_excel('台湾海峡和东海最优预测指标汇总.xlsx', sheet_name='EastChinaSea', header=[0, 1], index_col=0)

# 台湾海峡 个体模型和元模型最优预测指标
MLP_RMSE_TaiwanStrait = data_TaiwanStrait.values[0,0::2]
MLP_CE_TaiwanStrait = data_TaiwanStrait.values[0,1::2]
LSTM_RMSE_TaiwanStrait = data_TaiwanStrait.values[1,0::2]
LSTM_CE_TaiwanStrait = data_TaiwanStrait.values[1,1::2]
CNN_RMSE_TaiwanStrait = data_TaiwanStrait.values[2,0::2]
CNN_CE_TaiwanStrait = data_TaiwanStrait.values[2,1::2]
CNNLSTM_RMSE_TaiwanStrait = data_TaiwanStrait.values[3,0::2]
CNNLSTM_CE_TaiwanStrait = data_TaiwanStrait.values[3,1::2]
Stacking_RMSE_TaiwanStrait = data_TaiwanStrait.values[4,0::2]
Stacking_CE_TaiwanStrait = data_TaiwanStrait.values[4,1::2]
Persistent_RMSE_TaiwanStrait = data_TaiwanStrait.values[5,0::2]
Persistent_CE_TaiwanStrait = data_TaiwanStrait.values[5,1::2]

# 东海 个体模型和元模型最优预测指标
MLP_RMSE_EastChinaSea = data_EastChinaSea.values[0,0::2]
MLP_CE_EastChinaSea = data_EastChinaSea.values[0,1::2]
LSTM_RMSE_EastChinaSea = data_EastChinaSea.values[1,0::2]
LSTM_CE_EastChinaSea = data_EastChinaSea.values[1,1::2]
CNN_RMSE_EastChinaSea = data_EastChinaSea.values[2,0::2]
CNN_CE_EastChinaSea = data_EastChinaSea.values[2,1::2]
CNNLSTM_RMSE_EastChinaSea = data_EastChinaSea.values[3,0::2]
CNNLSTM_CE_EastChinaSea = data_EastChinaSea.values[3,1::2]
Stacking_RMSE_EastChinaSea = data_EastChinaSea.values[4,0::2]
Stacking_CE_EastChinaSea = data_EastChinaSea.values[4,1::2]
Persistent_RMSE_EastChinaSea = data_EastChinaSea.values[5,0::2]
Persistent_CE_EastChinaSea = data_EastChinaSea.values[5,1::2]

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
lead_time = [1, 3, 5, 7]
labels = ['MLP', 'LSTM', 'CNN', 'CNNLSTM', 'Stacking', 'Persist']
# RMSE for Taiwan Strait
axs[0, 0].set_xlim(0.5, 7.5)
axs[0, 0].plot(lead_time, MLP_RMSE_TaiwanStrait, 'b+-', linewidth=1.5, markersize=10)
axs[0, 0].plot(lead_time, LSTM_RMSE_TaiwanStrait, 'rx-', linewidth=1.5, markersize=10)
axs[0, 0].plot(lead_time, CNN_RMSE_TaiwanStrait, 'go-', linewidth=1.5, markersize=10)
axs[0, 0].plot(lead_time, CNNLSTM_RMSE_TaiwanStrait, 'y*-', linewidth=1.5, markersize=10)
axs[0, 0].plot(lead_time, Stacking_RMSE_TaiwanStrait, 'ks-', linewidth=1.5, markersize=10)
axs[0, 0].plot(lead_time, Persistent_RMSE_TaiwanStrait, 'm^-', linewidth=1.5, markersize=10)
axs[0, 0].set_title('(a)', fontsize=20, weight='bold', loc='center', y=-0.2)
axs[0, 0].set_xlabel('Forecasting horizons (day)', fontsize=20)
axs[0, 0].set_ylabel('RMSE (℃)', fontsize=20)
# axs[0, 0].xaxis.set_major_locator(MultipleLocator(2))
axs[0, 0].set_xticks(ticks=lead_time)
axs[0, 0].tick_params(labelsize=20)
axs[0, 0].grid(linestyle='--', linewidth=0.5)
axs[0, 0].legend(loc='best', labels=labels, prop={'size': 16})
# CE for Taiwan Strait
axs[0, 1].set_xlim(0.5, 7.5)
axs[0, 1].plot(lead_time, MLP_CE_TaiwanStrait, 'b+-', linewidth=1.5, markersize=10)
axs[0, 1].plot(lead_time, LSTM_CE_TaiwanStrait, 'rx-', linewidth=1.5, markersize=10)
axs[0, 1].plot(lead_time, CNN_CE_TaiwanStrait, 'go-', linewidth=1.5, markersize=10)
axs[0, 1].plot(lead_time, CNNLSTM_CE_TaiwanStrait, 'y*-', linewidth=1.5, markersize=10)
axs[0, 1].plot(lead_time, Stacking_CE_TaiwanStrait, 'ks-', linewidth=1.5, markersize=10)
axs[0, 1].plot(lead_time, Persistent_CE_TaiwanStrait, 'm^-', linewidth=1.5, markersize=10)
axs[0, 1].set_title('(b)', fontsize=20, weight='bold', loc='center', y=-0.2)
axs[0, 1].set_xlabel('Forecasting horizons (day)', fontsize=20)
axs[0, 1].set_ylabel('CE', fontsize=20)
# axs[0, 0].xaxis.set_major_locator(MultipleLocator(2))
axs[0, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
axs[0, 1].set_xticks(ticks=lead_time)
axs[0, 1].tick_params(labelsize=20)
axs[0, 1].grid(linestyle='--', linewidth=0.5)
axs[0, 1].legend(loc='best', labels=labels, prop={'size': 16})
# RMSE for East China Sea
axs[1, 0].set_xlim(0.5, 7.5)
axs[1, 0].plot(lead_time, MLP_RMSE_EastChinaSea, 'b+-', linewidth=1.5, markersize=10)
axs[1, 0].plot(lead_time, LSTM_RMSE_EastChinaSea, 'rx-', linewidth=1.5, markersize=10)
axs[1, 0].plot(lead_time, CNN_RMSE_EastChinaSea, 'go-', linewidth=1.5, markersize=10)
axs[1, 0].plot(lead_time, CNNLSTM_RMSE_EastChinaSea, 'y*-', linewidth=1.5, markersize=10)
axs[1, 0].plot(lead_time, Stacking_RMSE_EastChinaSea, 'ks-', linewidth=1.5, markersize=10)
axs[1, 0].plot(lead_time, Persistent_RMSE_EastChinaSea, 'm^-', linewidth=1.5, markersize=10)
axs[1, 0].set_title('(c)', fontsize=20, weight='bold', loc='center', y=-0.2)
axs[1, 0].set_xlabel('Forecasting horizons (day)', fontsize=20)
axs[1, 0].set_ylabel('RMSE (℃)', fontsize=20)
# axs[0, 0].xaxis.set_major_locator(MultipleLocator(2))
axs[1, 0].set_xticks(ticks=lead_time)
axs[1, 0].tick_params(labelsize=20)
axs[1, 0].grid(linestyle='--', linewidth=0.5)
axs[1, 0].legend(loc='best', labels=labels, prop={'size': 16})
# CE for East China Sea
axs[1, 1].set_xlim(0.5, 7.5)
axs[1, 1].plot(lead_time, MLP_CE_EastChinaSea, 'b+-', linewidth=1.5, markersize=10)
axs[1, 1].plot(lead_time, LSTM_CE_EastChinaSea, 'rx-', linewidth=1.5, markersize=10)
axs[1, 1].plot(lead_time, CNN_CE_EastChinaSea, 'go-', linewidth=1.5, markersize=10)
axs[1, 1].plot(lead_time, CNNLSTM_CE_EastChinaSea, 'y*-', linewidth=1.5, markersize=10)
axs[1, 1].plot(lead_time, Stacking_CE_EastChinaSea, 'ks-', linewidth=1.5, markersize=10)
axs[1, 1].plot(lead_time, Persistent_CE_EastChinaSea, 'm^-', linewidth=1.5, markersize=10)
axs[1, 1].set_title('(d)', fontsize=20, weight='bold', loc='center', y=-0.2)
axs[1, 1].set_xlabel('Forecasting horizons (day)', fontsize=20)
axs[1, 1].set_ylabel('CE', fontsize=20)
# axs[0, 0].xaxis.set_major_locator(MultipleLocator(2))
axs[1, 1].set_xticks(ticks=lead_time)
axs[1, 1].tick_params(labelsize=20)
axs[1, 1].grid(linestyle='--', linewidth=0.5)
axs[1, 1].legend(loc='best', labels=labels, prop={'size': 16})

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('figure3.jpg', dpi=300)
plt.show()
print('OK')