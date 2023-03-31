

# 保存子模型CNN-LSTM用于堆叠泛化

from numpy import argwhere, array, meshgrid
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
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

# 拟合预测SST的CNN-LSTM模型
def fit_CNNLSTM_model(X_train, y_train, filters, kernel_size, n_steps, n_neurons, n_inputs, n_layers, dropout_rate, n_outputs, epochs, batch_size):
    model = Sequential()
    # 1
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=1, activation='relu', padding='same',
                     data_format='channels_last', kernel_initializer='glorot_uniform', input_shape=(n_steps, n_inputs)))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
    model.add(Dropout(dropout_rate))
    # 1
    # # 2
    for i in range(n_layers-1):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=1, activation='relu', padding='same',
                         data_format='channels_last', kernel_initializer='glorot_uniform'))
        model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
        model.add(Dropout(dropout_rate))
    # # 2

    # model.add(Flatten())
    # model.add(Dense(50, activation='relu'))
    model.add(LSTM(n_neurons, activation='relu', return_sequences=False, kernel_initializer='glorot_uniform'))
    model.add(Dense(n_outputs, kernel_initializer='glorot_uniform'))
    model.summary()
    model.compile(optimizer='Adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.5, patience=10)
    curtime = strftime('%Y%m%d%H%M%S', localtime(time()))
    temp = curtime + '_CNNLSTM_best_model.h5'
    mc = ModelCheckpoint(filepath=temp, monitor='val_loss', verbose=1, save_best_only=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                        verbose=1, callbacks=[mc, es, reduce_lr], shuffle=False)
    return model

lead_time = 1
# 参数
n_steps = 4
n_outputs = 1
batch_size = 128
n_neurons = 30
filters = 128
# filters2 = [1]
dropout_rate = 0.0
n_layers = 1
epochs = 400
kernel_size = 3

# f = h5py.File('raw_train.h5', 'r')
# raw_train = f.get('raw_train')[:]
# f = h5py.File('raw_test.h5', 'r')
# raw_test = f.get('raw_test')[:]

f = h5py.File('donghai2_raw_train.h5', 'r')
raw_train = f.get('donghai2_raw_train')[:]
f = h5py.File('donghai2_raw_test.h5', 'r')
raw_test = f.get('donghai2_raw_test')[:]


scaler = StandardScaler()
# Conv1D要求每个样本的输入部分的形状是3D。[samples,timesteps,features]
# 所以要将raw_train展平，那么raw_train_scaled的形状是[samples,row*col]
raw_train_scaled = scaler.fit_transform(raw_train.reshape(-1, raw_train.shape[1] * raw_train.shape[2]))
raw_test_scaled = scaler.transform(raw_test.reshape(-1, raw_test.shape[1] * raw_test.shape[2]))
n_rows = raw_train.shape[1]
n_cols = raw_train.shape[2]

# 将SST时空序列数据转换为 监督学习问题
# X_train_valid和X_test的形状是[samples, n_steps, row*col], y_train_valid和y_test的形状是[samples,row*col]
X_train_valid, y_train_valid = create_data(raw_train_scaled, n_steps, lead_time)
X_test, y_test = create_data(raw_test_scaled, n_steps, lead_time)

# 训练数据建模，使用训练数据进行模型训练，展示训练数据和验证数据的MSE
n_inputs = X_train_valid.shape[2]
# X_train_valid = X_train_valid.reshape(X_train_valid.shape[0], n_inputs)
n_outputs = y_train_valid.shape[1]

validation_ratio = 0.5
len_train_valid = X_train_valid.shape[0]
X_train, y_train = X_train_valid[:int(len_train_valid*validation_ratio), :], y_train_valid[:int(len_train_valid*validation_ratio), :]
# X_valid, y_valid = X_train_valid[int(len_train_valid*validation_ratio):, :], y_train_valid[int(len_train_valid*validation_ratio):, :]

if path.exists('sub-models_CNNLSTM'):
    rmtree('sub-models_CNNLSTM')
makedirs('sub-models_CNNLSTM')

n_members = 100
rmse_test = []
for i in range(n_members):
    model = fit_CNNLSTM_model(X_train, y_train, filters, kernel_size, n_steps, n_neurons, n_inputs, n_layers, dropout_rate, n_outputs, epochs, batch_size)
    filename = 'sub-models_CNNLSTM/model_CNNLSTM_' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)
    yhat_test = model.predict(X_test)
    inv_yhat_test = scaler.inverse_transform(yhat_test)
    inv_y_test = scaler.inverse_transform(y_test)
    rmse_temp = mean_squared_error(inv_y_test, inv_yhat_test, squared=False)
    rmse_test.append(rmse_temp)
    print('rmse_test: %.3f ℃' % rmse_temp)

dt = DataFrame(rmse_test)
dt.to_csv('CNNLSTM_rmse_test.csv', index=False, header=False)
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
