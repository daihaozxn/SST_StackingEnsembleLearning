

# 保存子模型MLP用于堆叠泛化

from numpy import argwhere, array, meshgrid
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, ConvLSTM2D, Conv1D, BatchNormalization, MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
# from math import sqrt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from time import time, strftime, localtime
from tensorflow.keras.models import load_model
import ipykernel, h5py
from os import makedirs, path
from pandas import DataFrame
from shutil import rmtree

# 带时间步长的数据
def create_data(dataset, n_steps, lead_time):
    dataset_len = len(dataset)
    dataX, dataY = [], []
    for i in range(dataset_len - n_steps - lead_time + 1):
        tempX, tempY = dataset[i : (i+n_steps), :], dataset[i+n_steps-1+lead_time, :]
        dataX.append(tempX)
        dataY.append(tempY)
    return array(dataX), array(dataY)

# 拟合预测SST的MLP模型
def fit_MLP_model(X_train, y_train, n_neurons, n_inputs, dropout_rate, n_outputs, epochs, batch_size):
    model = Sequential()
    model.add(Dense(n_neurons, activation='relu', kernel_initializer='glorot_uniform', input_dim=n_inputs))  # 一层隐藏层
    model.add(Dropout(dropout_rate))
    if n_layers > 1:
        for i in range(n_layers - 1):
            model.add(Dense(n_neurons, activation='relu', kernel_initializer='glorot_uniform'))
            model.add(Dropout(dropout_rate))
    model.add(Dense(n_outputs))
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=500)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.5, patience=200)
    curtime = strftime('%Y%m%d%H%M%S', localtime(time()))
    temp = curtime + '_MLP_best_model.h5'
    mc = ModelCheckpoint(filepath=temp, monitor='val_loss', verbose=1, save_best_only=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                        verbose=1, callbacks=[mc, es, reduce_lr], shuffle=False)
    return model

lead_time = 1
# 参数
n_steps = 4
# n_outputs = 1
batch_size = 64
n_neurons = 100
dropout_rate = 0.0
n_layers = 1
epochs = 2000

f = h5py.File('donghai2_raw_train.h5', 'r')
raw_train = f.get('donghai2_raw_train')[:]
f = h5py.File('donghai2_raw_test.h5', 'r')
raw_test = f.get('donghai2_raw_test')[:]

scaler = StandardScaler()
# 将SST时空序列数据转换为 监督学习问题
# X_train_valid和X_test的形状是[samples, n_steps, row*col], y_train_valid和y_test的形状是[samples,row*col]
raw_train_scaled = scaler.fit_transform(raw_train.reshape(-1, raw_train.shape[1] * raw_train.shape[2]))
raw_test_scaled = scaler.transform(raw_test.reshape(-1, raw_test.shape[1] * raw_test.shape[2]))

# 将SST时空序列数据转换为 监督学习问题      y_test的形状是[samples,row*col]
X_train_valid, y_train_valid = create_data(raw_train_scaled, n_steps, lead_time)
X_test, y_test = create_data(raw_test_scaled, n_steps, lead_time)

# 训练数据建模，使用训练数据进行模型训练，展示训练数据和验证数据的MSE
n_inputs = X_train_valid.shape[1]*X_train_valid.shape[2]
X_train_valid = X_train_valid.reshape(X_train_valid.shape[0], n_inputs)
n_outputs = y_train_valid.shape[1]

validation_ratio = 0.5
len_train_valid = X_train_valid.shape[0]
X_train, y_train = X_train_valid[:int(len_train_valid*validation_ratio), :], y_train_valid[:int(len_train_valid*validation_ratio), :]
# X_valid, y_valid = X_train_valid[int(len_train_valid*validation_ratio):, :], y_train_valid[int(len_train_valid*validation_ratio):, :]
X_test = X_test.reshape(X_test.shape[0], n_inputs)

if path.exists('sub-models_MLP'):
    rmtree('sub-models_MLP')
makedirs('sub-models_MLP')

n_members = 50
rmse_test = []
for i in range(n_members):
    model = fit_MLP_model(X_train, y_train, n_neurons, n_inputs, dropout_rate, n_outputs, epochs, batch_size)
    filename = 'sub-models_MLP/model_MLP_' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)
    yhat_test = model.predict(X_test)
    inv_yhat_test = scaler.inverse_transform(yhat_test)
    inv_y_test = scaler.inverse_transform(y_test)
    rmse_temp = mean_squared_error(inv_y_test, inv_yhat_test, squared=False)
    rmse_test.append(rmse_temp)
    print('rmse_test: %.3f ℃' % rmse_temp)

dt = DataFrame(rmse_test)
dt.to_csv('MLP_rmse_test.csv', index=False, header=False)
# dt.to_csv('rmse_test1.csv', index=True, header='rmse')

# plt.figure()
# plt.boxplot(rmse_test, showmeans=True)
# plt.show()

# # 反标准化预测结果
# # inverse_transform的输入形状要求是二维张量 (n_samples, n_features)
# inv_yhat_test = scaler.inverse_transform(yhat_test)
# inv_y_test = scaler.inverse_transform(y_test)
# # print((inv_y_test.reshape(-1, raw_test.shape[1], raw_test.shape[2]) == raw_test[n_steps,:,:]).all())的结果是   true
#
# # 计算测试集中海域预测的RMSE
# rmse_test = mean_squared_error(inv_y_test, inv_yhat_test, squared=False)
# print('rmse_test: %.3f ℃' % rmse_test)
#
# plt.figure()
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='valid')
# plt.legend()
# plt.show()
