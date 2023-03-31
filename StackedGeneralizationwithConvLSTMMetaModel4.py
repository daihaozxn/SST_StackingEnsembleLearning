# 堆叠泛化
# 以ConvLSTM作为元学习器进行堆叠泛化
# 此程序导入的 数据 raw_train.h5和raw_test.h5 是训练38年(即1982-2019年)，测试1年(即2020年)
# 在train数据集上训练0级学习器(MLP,LSTM,CNN,CNNLSTM)，在validation数据集上训练1级学习器(ConvLSTM)，在test数据集上评估0级和1级学习器


from numpy import dstack
from numpy import argwhere, array, meshgrid
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, ConvLSTM2D, Conv2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
# from math import sqrt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from time import time, strftime, localtime
from tensorflow.keras.models import load_model
import ipykernel, h5py
from pandas import DataFrame

# 定义1级模型ConvLSTM
def TaiwanStraitSSTPredictionwithConvLSTM(n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs):
    model = Sequential()
    if n_layers == 1:
        # 将X的形状变为 [samples, n_steps, rows, columns, features]
        model.add(
            ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same',
                       data_format='channels_last', kernel_initializer='glorot_uniform',
                       input_shape=(n_steps, n_rows, n_cols, 1)))  # 一层隐藏层
        # model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    else:
        model.add(
            ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same',
                       data_format='channels_last', kernel_initializer='glorot_uniform',
                       input_shape=(n_steps, n_rows, n_cols, 1), return_sequences=True))  # 一层隐藏层
        # model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    if n_layers > 1:
        for i in range(n_layers - 2):
            model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu',
                                 padding='same',
                                 data_format='channels_last', kernel_initializer='glorot_uniform',
                                 return_sequences=True))
            # model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(
            ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu', padding='same',
                       data_format='channels_last', kernel_initializer='glorot_uniform',
                       return_sequences=False))
        # model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    # model.add(Dense(n_outputs))
    # model.summary()
    # model.add(Conv2D(filters=filters2, kernel_size=kernel_size2, strides=(1, 1), activation='relu', padding='same',
    #                  data_format='channels_last', kernel_initializer='glorot_uniform'))
    # model.add(Dense(30, kernel_initializer='he_uniform', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(n_outputs, kernel_initializer='glorot_uniform'))
    model.summary()
    model.compile(optimizer='Adam', loss='mse')
    return model

# load models from file   加载保存的子模型
# def load_all_models():
#     #all_models = list()
#     # for i in range(n_models):
#     #     # define filename for this ensemble
#     #     filename = 'sub-models/model_CNN_' + str(i + 1) + '.h5'
#     #     # load model from file
#     #     model = load_model(filename)
#     #     # add to list of members
#     #     all_models.append(model)
#     #     print('>loaded %s' % filename)
#     model_MLP = load_model('Submodels/model_MLP_3.h5')
#     model_LSTM = load_model('Submodels/model_LSTM_8.h5')
#     model_CNN = load_model('Submodels/model_CNN_2.h5')
#     model_CNNLSTM = load_model('Submodels/model_CNNLSTM_9.h5')
#
#     all_models = [model_MLP, model_LSTM, model_CNN, model_CNNLSTM]
#     return all_models

# # create stacked model input dataset as outputs from the ensemble  使用各子模型的预测输出作为1级模型的训练数据集
# def stacked_dataset(members, inputX):
#     stackX = None
#     for model in members:
#         # make prediction
#         yhat = model.predict(inputX, verbose=0)
#         # stack predictions into [rows, members, probabilities]
#         if stackX is None:
#             stackX = yhat
#         else:
#             stackX = dstack((stackX, yhat))
#     # flatten predictions to [rows, members x probabilities]
#     # stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
#     # stackX 的形状是 [samples, members, features]
#     return stackX

# fit a model based on the outputs from the ensemble members   拟合堆叠集合后的1级模型
def fit_stacked_model(stackedX, inputy, n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs):
    # create dataset using ensemble
    # stackedX = stacked_dataset(members, inputX)
    stackedX = stackedX.reshape(stackedX.shape[0], stackedX.shape[2], n_rows, n_cols, 1)
    # fit stacked model
    model = TaiwanStraitSSTPredictionwithConvLSTM(n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.5, patience=10)
    curtime = strftime('%Y%m%d%H%M%S', localtime(time()))
    temp = curtime + 'best_model.h5'
    mc = ModelCheckpoint(filepath=temp, monitor='val_loss', verbose=1, save_best_only=True)

    model.fit(stackedX, inputy, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                        verbose=1, callbacks=[mc, es, reduce_lr], shuffle=False)
    return model

# make a prediction with the stacked model 使用堆叠后的模型（1级模型）进行预测
def stacked_prediction(stackedX, model, n_rows, n_cols):
    # create dataset using ensemble
    # stackedX = stacked_dataset(members, inputX)
    stackedX = stackedX.reshape(stackedX.shape[0], stackedX.shape[2], n_rows, n_cols, 1)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

# 带时间步长的数据
def create_data(dataset, n_steps, lead_time):
    dataset_len = len(dataset)
    dataX, dataY = [], []
    for i in range(dataset_len - n_steps - lead_time + 1):
        tempX, tempY = dataset[i: (i + n_steps), :], dataset[i + n_steps - 1 + lead_time, :]
        dataX.append(tempX)
        dataY.append(tempY)
    return array(dataX), array(dataY)

lead_time = 1
# 参数
n_steps = 4
n_outputs = 1
batch_size = 512
# n_neurons = [30, 20, 10]
filters = 128
# filters2 = [1]
dropout_rate = 0.0
n_layers = 1
epochs = 50
kernel_size = (3, 3)

f = h5py.File('raw_train.h5', 'r')
raw_train = f.get('raw_train')[:]
f = h5py.File('raw_test.h5', 'r')
raw_test = f.get('raw_test')[:]

scaler = StandardScaler()
# ConvLSTM2D要求每个样本的输入部分的形状是5D。[samples,timesteps,rows,columns,features]
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
X_train_valid = X_train_valid.reshape(X_train_valid.shape[0], n_steps, n_rows, n_cols, 1)
y_train_valid = y_train_valid.reshape(y_train_valid.shape[0], n_rows, n_cols, 1)
X_test = X_test.reshape(X_test.shape[0], n_steps, n_rows, n_cols, 1)
y_test = y_test.reshape(y_test.shape[0], n_rows, n_cols, 1)

validation_ratio = 0.5
len_train_valid = X_train_valid.shape[0]
# X_train, y_train = X_train_valid[:int(len_train_valid * validation_ratio), :], y_train_valid[:int(len_train_valid * validation_ratio), :]
X_valid, y_valid = X_train_valid[int(len_train_valid * validation_ratio):, :], y_train_valid[int(len_train_valid * validation_ratio):, :]

X_test_MLP = X_test.reshape(-1, n_steps*n_rows*n_cols)
X_test_LSTM = X_test.reshape(-1, n_steps, n_rows*n_cols)
X_test_CNN = X_test.reshape(-1, n_steps, n_rows*n_cols)
X_test_CNNLSTM = X_test.reshape(-1, n_steps, n_rows*n_cols)

# load all models  加载保存的子模型
# n_members = 4
# members = load_all_models()
# print('Loaded %d models' % len(members))
model_MLP = load_model('Submodels/model_MLP_3.h5')
model_LSTM = load_model('Submodels/model_LSTM_8.h5')
model_CNN = load_model('Submodels/model_CNN_2.h5')
model_CNNLSTM = load_model('Submodels/model_CNNLSTM_9.h5')

# # evaluate standalone models on test dataset  评估单个子模型在测试数据集上的表现
# for model in members:
#     # loss, _ = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
#     # print('Sub-Model loss: %.3f' % loss)
#     yhat_test = model.predict(X_test_CNN)
#     # 反标准化预测结果
#     # inverse_transform的输入形状要求是二维张量 (n_samples, n_features)
#     inv_yhat_test = scaler.inverse_transform(yhat_test)
#     inv_y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[1] * y_test.shape[2] * y_test.shape[3]))
#     # print((inv_y_test.reshape(-1, raw_test.shape[1], raw_test.shape[2]) == raw_test[n_steps,:,:]).all())的结果是   true
#
#     # 计算每个 子模型 对测试集海域预测的RMSE
#     rmse_test = mean_squared_error(inv_y_test, inv_yhat_test, squared=False)
#     print('rmse_test: %.3f ℃' % rmse_test)

# evaluate standalone models on test dataset  评估单个子模型在测试数据集上的表现
# 测试标签
inv_y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[1] * y_test.shape[2] * y_test.shape[3]))
# MLP
yhat_test_MLP = model_MLP.predict(X_test_MLP)
inv_yhat_test_MLP = scaler.inverse_transform(yhat_test_MLP)
rmse_test_MLP = mean_squared_error(inv_y_test, inv_yhat_test_MLP, squared=False)
# print('rmse_test_MLP: %.3f ℃' % rmse_test_MLP)
# LSTM
yhat_test_LSTM = model_LSTM.predict(X_test_LSTM)
inv_yhat_test_LSTM = scaler.inverse_transform(yhat_test_LSTM)
rmse_test_LSTM = mean_squared_error(inv_y_test, inv_yhat_test_LSTM, squared=False)
# print('rmse_test_LSTM: %.3f ℃' % rmse_test_LSTM)
# CNN
yhat_test_CNN = model_CNN.predict(X_test_CNN)
inv_yhat_test_CNN = scaler.inverse_transform(yhat_test_CNN)
rmse_test_CNN = mean_squared_error(inv_y_test, inv_yhat_test_CNN, squared=False)
# print('rmse_test_CNN: %.3f ℃' % rmse_test_CNN)
# CNNLSTM
yhat_test_CNNLSTM = model_CNNLSTM.predict(X_test_CNNLSTM)
inv_yhat_test_CNNLSTM = scaler.inverse_transform(yhat_test_CNNLSTM)
rmse_test_CNNLSTM = mean_squared_error(inv_y_test, inv_yhat_test_CNNLSTM, squared=False)
# print('rmse_test_CNNLSTM: %.3f ℃' % rmse_test_CNNLSTM)

# 对各个0级模型的测试集预测结果做平均，作为最终的预测结果，然后计算rmse
yhat_test_ModelsAve = (yhat_test_MLP + yhat_test_LSTM + yhat_test_CNN + yhat_test_CNNLSTM)/4
inv_yhat_test_ModelsAve = scaler.inverse_transform(yhat_test_ModelsAve)
rmse_test_ModelsAve = mean_squared_error(inv_y_test, inv_yhat_test_ModelsAve, squared=False)
# print('rmse_test_ModelsAve: %.3f ℃' % rmse_test_ModelsAve)

# 根据各个0级模型预测结果的rmses计算逆权重，加权平均作为最终的预测结果，然后计算rmse
# weights_test_MLP = 1 -

# create stacked model input dataset as outputs from the ensemble  使用各子模型在验证集上的预测输出作为1级模型的训练数据集
X_valid_MLP = X_valid.reshape(-1, n_steps*n_rows*n_cols)
X_valid_LSTM = X_valid.reshape(-1, n_steps, n_rows*n_cols)
X_valid_CNN = X_valid.reshape(-1, n_steps, n_rows*n_cols)
X_valid_CNNLSTM = X_valid.reshape(-1, n_steps, n_rows*n_cols)
# MLP
yhat_valid_MLP = model_MLP.predict(X_valid_MLP)
# LSTM
yhat_valid_LSTM = model_LSTM.predict(X_valid_LSTM)
# CNN
yhat_valid_CNN = model_CNN.predict(X_valid_CNN)
# CNNLSTM
yhat_valid_CNNLSTM = model_CNNLSTM.predict(X_valid_CNNLSTM)

stackX_valid = dstack((yhat_valid_MLP, yhat_valid_LSTM, yhat_valid_CNN, yhat_valid_CNNLSTM))

# fit stacked model using the ensemble  拟合堆叠集合后的1级模型
model = fit_stacked_model(stackX_valid, y_valid, n_layers, n_steps, n_rows, n_cols, dropout_rate, filters, kernel_size, n_outputs)

# evaluate model on test set 评估堆叠后模型在测试集上的表现
X_test_MLP = X_test.reshape(-1, n_steps*n_rows*n_cols)
X_test_LSTM = X_test.reshape(-1, n_steps, n_rows*n_cols)
X_test_CNN = X_test.reshape(-1, n_steps, n_rows*n_cols)
X_test_CNNLSTM = X_test.reshape(-1, n_steps, n_rows*n_cols)
# MLP
yhat_test_MLP = model_MLP.predict(X_test_MLP)
# LSTM
yhat_test_LSTM = model_LSTM.predict(X_test_LSTM)
# CNN
yhat_test_CNN = model_CNN.predict(X_test_CNN)
# CNNLSTM
yhat_test_CNNLSTM = model_CNNLSTM.predict(X_test_CNNLSTM)

stackX_test = dstack((yhat_test_MLP, yhat_test_LSTM, yhat_test_CNN, yhat_test_CNNLSTM))

yhat = stacked_prediction(stackX_test, model, n_rows, n_cols)
# 反标准化预测结果
# inverse_transform的输入形状要求是二维张量 (n_samples, n_features)
inv_yhat = scaler.inverse_transform(yhat.reshape(-1, yhat.shape[1] * yhat.shape[2] * yhat.shape[3]))
# inv_y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[1] * y_test.shape[2] * y_test.shape[3]))
# print((inv_y_test.reshape(-1, raw_test.shape[1], raw_test.shape[2]) == raw_test[n_steps,:,:]).all())的结果是   true

# 计算1级模型 对测试集海域预测的RMSE
rmse_test = mean_squared_error(inv_y_test, inv_yhat, squared=False)
# print('rmse_test: %.3f ℃' % rmse_test)

# 存各个 rmse
# rmse = [rmse_test_MLP, rmse_test_LSTM, rmse_test_CNN, rmse_test_CNNLSTM, rmse_test_ModelsAve, rmse_test]
rmse = {'MLP': [rmse_test_MLP],
        'LSTM': [rmse_test_LSTM],
        'CNN': [rmse_test_CNN],
        'CNNLSTM': [rmse_test_CNNLSTM],
        'Ave': [rmse_test_ModelsAve],
        'Stacking': [rmse_test]}
dt_rmse = DataFrame(rmse)
dt_rmse.to_csv('rmse.csv', index=False, header=True)