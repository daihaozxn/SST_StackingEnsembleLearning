# 在一个图上画出 两个海域的 RMSE CE 的空间分布


import matplotlib.pyplot as plt
from pandas import read_excel
import ipykernel, h5py
from sklearn.preprocessing import StandardScaler
from numpy import array, dstack, min, max, sqrt, mean, sum
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from matplotlib import colors


# 带时间步长的数据
def create_data(dataset, n_steps, lead_time):
    dataset_len = len(dataset)
    dataX, dataY = [], []
    for i in range(dataset_len - n_steps - lead_time + 1):
        tempX, tempY = dataset[i : (i+n_steps), :], dataset[i+n_steps-1+lead_time, :]
        dataX.append(tempX)
        dataY.append(tempY)
    return array(dataX), array(dataY)

def MeticsSpatialDistributionShow3(lead_time):
    n_steps = 4

    f_tw = h5py.File('raw_train.h5', 'r')
    raw_train_tw = f_tw.get('raw_train')[:]
    f_tw = h5py.File('raw_test.h5', 'r')
    raw_test_tw = f_tw.get('raw_test')[:]

    f_dh = h5py.File('donghai_raw_train.h5', 'r')
    raw_train_dh = f_dh.get('donghai_raw_train')[:]
    f_dh = h5py.File('donghai_raw_test.h5', 'r')
    raw_test_dh = f_dh.get('donghai_raw_test')[:]

    if lead_time == 1:
        model_MLP_tw = load_model('Submodels/lead_time_1/model_MLP_16.h5')
        model_LSTM_tw = load_model('Submodels/lead_time_1/model_LSTM_68.h5')
        model_CNN_tw = load_model('Submodels/lead_time_1/model_CNN_47.h5')
        model_CNNLSTM_tw = load_model('Submodels/lead_time_1/model_CNNLSTM_94.h5')
        model_ConvLSTMStacking_tw = load_model('Submodels/lead_time_1/model_ConvLSTMStacking_58.h5')

        model_MLP_dh = load_model('Submodels/DH_lead_time_1/model_MLP_88.h5')
        model_LSTM_dh = load_model('Submodels/DH_lead_time_1/model_LSTM_52.h5')
        model_CNN_dh = load_model('Submodels/DH_lead_time_1/model_CNN_69.h5')
        model_CNNLSTM_dh = load_model('Submodels/DH_lead_time_1/model_CNNLSTM_20.h5')
        model_ConvLSTMStacking_dh = load_model('Submodels/DH_lead_time_1/model_ConvLSTMStacking_60.h5')
    elif lead_time == 3:
        model_MLP_tw = load_model('Submodels/lead_time_3/model_MLP_17.h5')
        model_LSTM_tw = load_model('Submodels/lead_time_3/model_LSTM_31.h5')
        model_CNN_tw = load_model('Submodels/lead_time_3/model_CNN_94.h5')
        model_CNNLSTM_tw = load_model('Submodels/lead_time_3/model_CNNLSTM_56.h5')
        model_ConvLSTMStacking_tw = load_model('Submodels/lead_time_3/model_ConvLSTMStacking_27.h5')

        model_MLP_dh = load_model('Submodels/DH_lead_time_3/model_MLP_22.h5')
        model_LSTM_dh = load_model('Submodels/DH_lead_time_3/model_LSTM_47.h5')
        model_CNN_dh = load_model('Submodels/DH_lead_time_3/model_CNN_35.h5')
        model_CNNLSTM_dh = load_model('Submodels/DH_lead_time_3/model_CNNLSTM_5.h5')
        model_ConvLSTMStacking_dh = load_model('Submodels/DH_lead_time_3/model_ConvLSTMStacking_27.h5')
    elif lead_time == 5:
        model_MLP_tw = load_model('Submodels/lead_time_5/model_MLP_3.h5')
        model_LSTM_tw = load_model('Submodels/lead_time_5/model_LSTM_46.h5')
        model_CNN_tw = load_model('Submodels/lead_time_5/model_CNN_75.h5')
        model_CNNLSTM_tw = load_model('Submodels/lead_time_5/model_CNNLSTM_1.h5')
        model_ConvLSTMStacking_tw = load_model('Submodels/lead_time_5/model_ConvLSTMStacking_38.h5')

        model_MLP_dh = load_model('Submodels/DH_lead_time_5/model_MLP_36.h5')
        model_LSTM_dh = load_model('Submodels/DH_lead_time_5/model_LSTM_78.h5')
        model_CNN_dh = load_model('Submodels/DH_lead_time_5/model_CNN_27.h5')
        model_CNNLSTM_dh = load_model('Submodels/DH_lead_time_5/model_CNNLSTM_76.h5')
        model_ConvLSTMStacking_dh = load_model('Submodels/DH_lead_time_5/model_ConvLSTMStacking_43.h5')
    elif lead_time == 7:
        model_MLP_tw = load_model('Submodels/lead_time_7/model_MLP_36.h5')
        model_LSTM_tw = load_model('Submodels/lead_time_7/model_LSTM_5.h5')
        model_CNN_tw = load_model('Submodels/lead_time_7/model_CNN_87.h5')
        model_CNNLSTM_tw = load_model('Submodels/lead_time_7/model_CNNLSTM_12.h5')
        model_ConvLSTMStacking_tw = load_model('Submodels/lead_time_7/model_ConvLSTMStacking_78.h5')

        model_MLP_dh = load_model('Submodels/DH_lead_time_7/model_MLP_32.h5')
        model_LSTM_dh = load_model('Submodels/DH_lead_time_7/model_LSTM_2.h5')
        model_CNN_dh = load_model('Submodels/DH_lead_time_7/model_CNN_78.h5')
        model_CNNLSTM_dh = load_model('Submodels/DH_lead_time_7/model_CNNLSTM_59.h5')
        model_ConvLSTMStacking_dh = load_model('Submodels/DH_lead_time_7/model_ConvLSTMStacking_7.h5')

    scaler_tw = StandardScaler()
    # 所以要将raw_train展平，那么raw_train_scaled的形状是[samples,row*col]
    raw_train_scaled_tw = scaler_tw.fit_transform(raw_train_tw.reshape(-1, raw_train_tw.shape[1] * raw_train_tw.shape[2]))
    raw_test_scaled_tw = scaler_tw.transform(raw_test_tw.reshape(-1, raw_test_tw.shape[1] * raw_test_tw.shape[2]))
    n_rows = raw_test_tw.shape[1]
    n_cols = raw_test_tw.shape[2]

    # 将SST时空序列数据转换为 监督学习问题
    # X_train_valid和X_test的形状是[samples, n_steps, row*col], y_train_valid和y_test的形状是[samples,row*col]
    # X_train_valid, y_train_valid = create_data(raw_train_scaled, n_steps, lead_time)
    X_test_tw, y_test_tw = create_data(raw_test_scaled_tw, n_steps, lead_time)

    X_test_MLP_tw = X_test_tw.reshape(-1, n_steps*n_rows*n_cols)
    X_test_LSTM_tw = X_test_tw.reshape(-1, n_steps, n_rows*n_cols)
    X_test_CNN_tw = X_test_tw.reshape(-1, n_steps, n_rows*n_cols)
    X_test_CNNLSTM_tw = X_test_tw.reshape(-1, n_steps, n_rows*n_cols)
    # X_test_ConvLSTMStacking = X_test.reshape(-1, n_steps, n_rows, n_cols, 1)
    # MLP
    yhat_test_MLP_tw = model_MLP_tw.predict(X_test_MLP_tw)
    # LSTM
    yhat_test_LSTM_tw = model_LSTM_tw.predict(X_test_LSTM_tw)
    # CNN
    yhat_test_CNN_tw = model_CNN_tw.predict(X_test_CNN_tw)
    # CNNLSTM
    yhat_test_CNNLSTM_tw = model_CNNLSTM_tw.predict(X_test_CNNLSTM_tw)

    stackX_test_tw = dstack((yhat_test_MLP_tw, yhat_test_LSTM_tw, yhat_test_CNN_tw, yhat_test_CNNLSTM_tw))
    stackX_test_tw = stackX_test_tw.reshape(stackX_test_tw.shape[0], stackX_test_tw.shape[2], n_rows, n_cols, 1)

    # ConvLSTMStacking
    yhat_test_ConvLSTMStacking_tw = model_ConvLSTMStacking_tw.predict(stackX_test_tw)

    inv_y_test_tw = scaler_tw.inverse_transform(y_test_tw)
    # MLP RMSE CE
    inv_yhat_test_MLP_tw = scaler_tw.inverse_transform(yhat_test_MLP_tw)
    rmse_test_MLP_Spatial_tw = sqrt(mean((inv_y_test_tw - inv_yhat_test_MLP_tw) ** 2, axis=0))
    ce_test_MLP_Spatial_tw = 1 - (sum((inv_y_test_tw - inv_yhat_test_MLP_tw) ** 2, axis=0)) / (sum((inv_y_test_tw - mean(inv_y_test_tw)) ** 2, axis=0))
    # LSTM RMSE CE
    inv_yhat_test_LSTM_tw = scaler_tw.inverse_transform(yhat_test_LSTM_tw)
    rmse_test_LSTM_Spatial_tw = sqrt(mean((inv_y_test_tw - inv_yhat_test_LSTM_tw) ** 2, axis=0))
    ce_test_LSTM_Spatial_tw = 1 - (sum((inv_y_test_tw - inv_yhat_test_LSTM_tw) ** 2, axis=0)) / (sum((inv_y_test_tw - mean(inv_y_test_tw)) ** 2, axis=0))
    # CNN RMSE CE
    inv_yhat_test_CNN_tw = scaler_tw.inverse_transform(yhat_test_CNN_tw)
    rmse_test_CNN_Spatial_tw = sqrt(mean((inv_y_test_tw - inv_yhat_test_CNN_tw) ** 2, axis=0))
    ce_test_CNN_Spatial_tw = 1 - (sum((inv_y_test_tw - inv_yhat_test_CNN_tw) ** 2, axis=0)) / (sum((inv_y_test_tw - mean(inv_y_test_tw)) ** 2, axis=0))
    # CNNLSTM RMSE CE
    inv_yhat_test_CNNLSTM_tw = scaler_tw.inverse_transform(yhat_test_CNNLSTM_tw)
    rmse_test_CNNLSTM_Spatial_tw = sqrt(mean((inv_y_test_tw - inv_yhat_test_CNNLSTM_tw) ** 2, axis=0))
    ce_test_CNNLSTM_Spatial_tw = 1 - (sum((inv_y_test_tw - inv_yhat_test_CNNLSTM_tw) ** 2, axis=0)) / (sum((inv_y_test_tw - mean(inv_y_test_tw)) ** 2, axis=0))
    # ConvLSTMStacking RMSE CE
    inv_yhat_test_ConvLSTMStacking_tw = scaler_tw.inverse_transform(yhat_test_ConvLSTMStacking_tw.reshape(-1, yhat_test_ConvLSTMStacking_tw.shape[1]*yhat_test_ConvLSTMStacking_tw.shape[2]*yhat_test_ConvLSTMStacking_tw.shape[3]))
    rmse_test_ConvLSTMStacking_Spatial_tw = sqrt(mean((inv_y_test_tw - inv_yhat_test_ConvLSTMStacking_tw) ** 2, axis=0))
    ce_test_ConvLSTMStacking_Spatial_tw = 1 - (sum((inv_y_test_tw - inv_yhat_test_ConvLSTMStacking_tw) ** 2, axis=0)) / (sum((inv_y_test_tw - mean(inv_y_test_tw)) ** 2, axis=0))

    vmin_rmse_tw = array([rmse_test_MLP_Spatial_tw, rmse_test_LSTM_Spatial_tw, rmse_test_CNN_Spatial_tw, rmse_test_CNNLSTM_Spatial_tw, rmse_test_ConvLSTMStacking_Spatial_tw]).min()
    vmax_rmse_tw = array([rmse_test_MLP_Spatial_tw, rmse_test_LSTM_Spatial_tw, rmse_test_CNN_Spatial_tw, rmse_test_CNNLSTM_Spatial_tw, rmse_test_ConvLSTMStacking_Spatial_tw]).max()
    norm_rmse_tw = colors.Normalize(vmin=vmin_rmse_tw, vmax=vmax_rmse_tw)

    vmin_ce_tw = array([ce_test_MLP_Spatial_tw, ce_test_LSTM_Spatial_tw, ce_test_CNN_Spatial_tw, ce_test_CNNLSTM_Spatial_tw, ce_test_ConvLSTMStacking_Spatial_tw]).min()
    vmax_ce_tw = array([ce_test_MLP_Spatial_tw, ce_test_LSTM_Spatial_tw, ce_test_CNN_Spatial_tw, ce_test_CNNLSTM_Spatial_tw, ce_test_ConvLSTMStacking_Spatial_tw]).max()
    norm_ce_tw = colors.Normalize(vmin=vmin_ce_tw, vmax=vmax_ce_tw)

    rmse_test_MLP_Spatial_tw = rmse_test_MLP_Spatial_tw.reshape(n_rows, n_cols)
    rmse_test_LSTM_Spatial_tw = rmse_test_LSTM_Spatial_tw.reshape(n_rows, n_cols)
    rmse_test_CNN_Spatial_tw = rmse_test_CNN_Spatial_tw.reshape(n_rows, n_cols)
    rmse_test_CNNLSTM_Spatial_tw = rmse_test_CNNLSTM_Spatial_tw.reshape(n_rows, n_cols)
    rmse_test_ConvLSTMStacking_Spatial_tw = rmse_test_ConvLSTMStacking_Spatial_tw.reshape(n_rows, n_cols)

    ce_test_MLP_Spatial_tw = ce_test_MLP_Spatial_tw.reshape(n_rows, n_cols)
    ce_test_LSTM_Spatial_tw = ce_test_LSTM_Spatial_tw.reshape(n_rows, n_cols)
    ce_test_CNN_Spatial_tw = ce_test_CNN_Spatial_tw.reshape(n_rows, n_cols)
    ce_test_CNNLSTM_Spatial_tw = ce_test_CNNLSTM_Spatial_tw.reshape(n_rows, n_cols)
    ce_test_ConvLSTMStacking_Spatial_tw = ce_test_ConvLSTMStacking_Spatial_tw.reshape(n_rows, n_cols)

    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'

    scaler_dh = StandardScaler()
    # 所以要将raw_train展平，那么raw_train_scaled的形状是[samples,row*col]
    raw_train_scaled_dh = scaler_dh.fit_transform(raw_train_dh.reshape(-1, raw_train_dh.shape[1] * raw_train_dh.shape[2]))
    raw_test_scaled_dh = scaler_dh.transform(raw_test_dh.reshape(-1, raw_test_dh.shape[1] * raw_test_dh.shape[2]))
    n_rows = raw_test_dh.shape[1]
    n_cols = raw_test_dh.shape[2]

    # 将SST时空序列数据转换为 监督学习问题
    # X_train_valid和X_test的形状是[samples, n_steps, row*col], y_train_valid和y_test的形状是[samples,row*col]
    # X_train_valid, y_train_valid = create_data(raw_train_scaled, n_steps, lead_time)
    X_test_dh, y_test_dh = create_data(raw_test_scaled_dh, n_steps, lead_time)

    X_test_MLP_dh = X_test_dh.reshape(-1, n_steps*n_rows*n_cols)
    X_test_LSTM_dh = X_test_dh.reshape(-1, n_steps, n_rows*n_cols)
    X_test_CNN_dh = X_test_dh.reshape(-1, n_steps, n_rows*n_cols)
    X_test_CNNLSTM_dh = X_test_dh.reshape(-1, n_steps, n_rows*n_cols)
    # X_test_ConvLSTMStacking = X_test.reshape(-1, n_steps, n_rows, n_cols, 1)
    # MLP
    yhat_test_MLP_dh = model_MLP_dh.predict(X_test_MLP_dh)
    # LSTM
    yhat_test_LSTM_dh = model_LSTM_dh.predict(X_test_LSTM_dh)
    # CNN
    yhat_test_CNN_dh = model_CNN_dh.predict(X_test_CNN_dh)
    # CNNLSTM
    yhat_test_CNNLSTM_dh = model_CNNLSTM_dh.predict(X_test_CNNLSTM_dh)

    stackX_test_dh = dstack((yhat_test_MLP_dh, yhat_test_LSTM_dh, yhat_test_CNN_dh, yhat_test_CNNLSTM_dh))
    stackX_test_dh = stackX_test_dh.reshape(stackX_test_dh.shape[0], stackX_test_dh.shape[2], n_rows, n_cols, 1)

    # ConvLSTMStacking
    yhat_test_ConvLSTMStacking_dh = model_ConvLSTMStacking_dh.predict(stackX_test_dh)

    inv_y_test_dh = scaler_dh.inverse_transform(y_test_dh)
    # MLP RMSE CE
    inv_yhat_test_MLP_dh = scaler_dh.inverse_transform(yhat_test_MLP_dh)
    rmse_test_MLP_Spatial_dh = sqrt(mean((inv_y_test_dh - inv_yhat_test_MLP_dh) ** 2, axis=0))
    ce_test_MLP_Spatial_dh = 1 - (sum((inv_y_test_dh - inv_yhat_test_MLP_dh) ** 2, axis=0)) / (sum((inv_y_test_dh - mean(inv_y_test_dh)) ** 2, axis=0))
    # LSTM RMSE CE
    inv_yhat_test_LSTM_dh = scaler_dh.inverse_transform(yhat_test_LSTM_dh)
    rmse_test_LSTM_Spatial_dh = sqrt(mean((inv_y_test_dh - inv_yhat_test_LSTM_dh) ** 2, axis=0))
    ce_test_LSTM_Spatial_dh = 1 - (sum((inv_y_test_dh - inv_yhat_test_LSTM_dh) ** 2, axis=0)) / (sum((inv_y_test_dh - mean(inv_y_test_dh)) ** 2, axis=0))
    # CNN RMSE CE
    inv_yhat_test_CNN_dh = scaler_dh.inverse_transform(yhat_test_CNN_dh)
    rmse_test_CNN_Spatial_dh = sqrt(mean((inv_y_test_dh - inv_yhat_test_CNN_dh) ** 2, axis=0))
    ce_test_CNN_Spatial_dh = 1 - (sum((inv_y_test_dh - inv_yhat_test_CNN_dh) ** 2, axis=0)) / (sum((inv_y_test_dh - mean(inv_y_test_dh)) ** 2, axis=0))
    # CNNLSTM RMSE CE
    inv_yhat_test_CNNLSTM_dh = scaler_dh.inverse_transform(yhat_test_CNNLSTM_dh)
    rmse_test_CNNLSTM_Spatial_dh = sqrt(mean((inv_y_test_dh - inv_yhat_test_CNNLSTM_dh) ** 2, axis=0))
    ce_test_CNNLSTM_Spatial_dh = 1 - (sum((inv_y_test_dh - inv_yhat_test_CNNLSTM_dh) ** 2, axis=0)) / (sum((inv_y_test_dh - mean(inv_y_test_dh)) ** 2, axis=0))
    # ConvLSTMStacking RMSE CE
    inv_yhat_test_ConvLSTMStacking_dh = scaler_dh.inverse_transform(yhat_test_ConvLSTMStacking_dh.reshape(-1, yhat_test_ConvLSTMStacking_dh.shape[1]*yhat_test_ConvLSTMStacking_dh.shape[2]*yhat_test_ConvLSTMStacking_dh.shape[3]))
    rmse_test_ConvLSTMStacking_Spatial_dh = sqrt(mean((inv_y_test_dh - inv_yhat_test_ConvLSTMStacking_dh) ** 2, axis=0))
    ce_test_ConvLSTMStacking_Spatial_dh = 1 - (sum((inv_y_test_dh - inv_yhat_test_ConvLSTMStacking_dh) ** 2, axis=0)) / (sum((inv_y_test_dh - mean(inv_y_test_dh)) ** 2, axis=0))

    vmin_rmse_dh = array([rmse_test_MLP_Spatial_dh, rmse_test_LSTM_Spatial_dh, rmse_test_CNN_Spatial_dh, rmse_test_CNNLSTM_Spatial_dh, rmse_test_ConvLSTMStacking_Spatial_dh]).min()
    vmax_rmse_dh = array([rmse_test_MLP_Spatial_dh, rmse_test_LSTM_Spatial_dh, rmse_test_CNN_Spatial_dh, rmse_test_CNNLSTM_Spatial_dh, rmse_test_ConvLSTMStacking_Spatial_dh]).max()
    norm_rmse_dh = colors.Normalize(vmin=vmin_rmse_dh, vmax=vmax_rmse_dh)

    vmin_ce_dh = array([ce_test_MLP_Spatial_dh, ce_test_LSTM_Spatial_dh, ce_test_CNN_Spatial_dh, ce_test_CNNLSTM_Spatial_dh, ce_test_ConvLSTMStacking_Spatial_dh]).min()
    vmax_ce_dh = array([ce_test_MLP_Spatial_dh, ce_test_LSTM_Spatial_dh, ce_test_CNN_Spatial_dh, ce_test_CNNLSTM_Spatial_dh, ce_test_ConvLSTMStacking_Spatial_dh]).max()
    norm_ce_dh = colors.Normalize(vmin=vmin_ce_dh, vmax=vmax_ce_dh)

    rmse_test_MLP_Spatial_dh = rmse_test_MLP_Spatial_dh.reshape(n_rows, n_cols)
    rmse_test_LSTM_Spatial_dh = rmse_test_LSTM_Spatial_dh.reshape(n_rows, n_cols)
    rmse_test_CNN_Spatial_dh = rmse_test_CNN_Spatial_dh.reshape(n_rows, n_cols)
    rmse_test_CNNLSTM_Spatial_dh = rmse_test_CNNLSTM_Spatial_dh.reshape(n_rows, n_cols)
    rmse_test_ConvLSTMStacking_Spatial_dh = rmse_test_ConvLSTMStacking_Spatial_dh.reshape(n_rows, n_cols)

    ce_test_MLP_Spatial_dh = ce_test_MLP_Spatial_dh.reshape(n_rows, n_cols)
    ce_test_LSTM_Spatial_dh = ce_test_LSTM_Spatial_dh.reshape(n_rows, n_cols)
    ce_test_CNN_Spatial_dh = ce_test_CNN_Spatial_dh.reshape(n_rows, n_cols)
    ce_test_CNNLSTM_Spatial_dh = ce_test_CNNLSTM_Spatial_dh.reshape(n_rows, n_cols)
    ce_test_ConvLSTMStacking_Spatial_dh = ce_test_ConvLSTMStacking_Spatial_dh.reshape(n_rows, n_cols)

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(18, 18))
    # MLP
    a = axs[0, 0].pcolormesh(rmse_test_MLP_Spatial_tw, norm=norm_rmse_tw, cmap='jet', shading='auto')
    axs[0, 0].xaxis.set_ticks([])
    axs[0, 0].yaxis.set_ticks([])
    axs[0, 0].set_title('MLP', fontsize=16, weight='bold', loc='center')
    axs[0, 0].set_ylabel('RMSE(℃) \n for Taiwan Strait', fontsize=16, weight='bold')
    # LSTM
    axs[0, 1].pcolormesh(rmse_test_LSTM_Spatial_tw, norm=norm_rmse_tw, cmap='jet', shading='auto')
    axs[0, 1].xaxis.set_ticks([])
    axs[0, 1].yaxis.set_ticks([])
    axs[0, 1].set_title('LSTM', fontsize=16, weight='bold', loc='center')
    # axs[0, 1].set_ylabel('RMSE', fontsize=16, weight='bold')
    # CNN
    axs[0, 2].pcolormesh(rmse_test_CNN_Spatial_tw, norm=norm_rmse_tw, cmap='jet', shading='auto')
    axs[0, 2].xaxis.set_ticks([])
    axs[0, 2].yaxis.set_ticks([])
    axs[0, 2].set_title('CNN', fontsize=16, weight='bold', loc='center')
    # CNNLSTM
    axs[0, 3].pcolormesh(rmse_test_CNNLSTM_Spatial_tw, norm=norm_rmse_tw, cmap='jet', shading='auto')
    axs[0, 3].xaxis.set_ticks([])
    axs[0, 3].yaxis.set_ticks([])
    axs[0, 3].set_title('CNNLSTM', fontsize=16, weight='bold', loc='center')
    # ConvLSTMStacking
    axs[0, 4].pcolormesh(rmse_test_ConvLSTMStacking_Spatial_tw, norm=norm_rmse_tw, cmap='jet', shading='auto')
    axs[0, 4].xaxis.set_ticks([])
    axs[0, 4].yaxis.set_ticks([])
    axs[0, 4].set_title('Stacking', fontsize=16, weight='bold', loc='center')
    fig.colorbar(a, ax=axs[0, :])

    # MLP
    b = axs[1, 0].pcolormesh(rmse_test_MLP_Spatial_dh, norm=norm_rmse_dh, cmap='jet', shading='auto')
    axs[1, 0].xaxis.set_ticks([])
    axs[1, 0].yaxis.set_ticks([])
    # axs[0, 0].set_title('MLP', fontsize=16, weight='bold', loc='center')
    axs[1, 0].set_ylabel('RMSE(℃) \n for East China Sea', fontsize=16, weight='bold')
    # LSTM
    axs[1, 1].pcolormesh(rmse_test_LSTM_Spatial_dh, norm=norm_rmse_dh, cmap='jet', shading='auto')
    axs[1, 1].xaxis.set_ticks([])
    axs[1, 1].yaxis.set_ticks([])
    # axs[1, 1].set_title('LSTM', fontsize=16, weight='bold', loc='center')
    # axs[0, 1].set_ylabel('RMSE', fontsize=16, weight='bold')
    # CNN
    axs[1, 2].pcolormesh(rmse_test_CNN_Spatial_dh, norm=norm_rmse_dh, cmap='jet', shading='auto')
    axs[1, 2].xaxis.set_ticks([])
    axs[1, 2].yaxis.set_ticks([])
    # axs[1, 2].set_title('CNN', fontsize=16, weight='bold', loc='center')
    # CNNLSTM
    axs[1, 3].pcolormesh(rmse_test_CNNLSTM_Spatial_dh, norm=norm_rmse_dh, cmap='jet', shading='auto')
    axs[1, 3].xaxis.set_ticks([])
    axs[1, 3].yaxis.set_ticks([])
    # axs[1, 3].set_title('CNNLSTM', fontsize=16, weight='bold', loc='center')
    # ConvLSTMStacking
    axs[1, 4].pcolormesh(rmse_test_ConvLSTMStacking_Spatial_dh, norm=norm_rmse_dh, cmap='jet', shading='auto')
    axs[1, 4].xaxis.set_ticks([])
    axs[1, 4].yaxis.set_ticks([])
    # axs[1, 4].set_title('Stacking', fontsize=16, weight='bold', loc='center')
    fig.colorbar(b, ax=axs[1, :])

    # MLP
    c = axs[2, 0].pcolormesh(ce_test_MLP_Spatial_tw, norm=norm_ce_tw, cmap='jet', shading='auto')
    axs[2, 0].xaxis.set_ticks([])
    axs[2, 0].yaxis.set_ticks([])
    # axs[1, 0].set_title('MLP', fontsize=16, weight='bold', loc='center')
    axs[2, 0].set_ylabel('CE \n for Taiwan Strait', fontsize=16, weight='bold')
    # LSTM
    axs[2, 1].pcolormesh(ce_test_LSTM_Spatial_tw, norm=norm_ce_tw, cmap='jet', shading='auto')
    axs[2, 1].xaxis.set_ticks([])
    axs[2, 1].yaxis.set_ticks([])
    # axs[1, 0].set_title('MLP', fontsize=16, weight='bold', loc='center')
    # axs[1, 1].set_ylabel('CE', fontsize=16, weight='bold')
    # CNN
    axs[2, 2].pcolormesh(ce_test_CNN_Spatial_tw, norm=norm_ce_tw, cmap='jet', shading='auto')
    axs[2, 2].xaxis.set_ticks([])
    axs[2, 2].yaxis.set_ticks([])
    # CNNLSTM
    axs[2, 3].pcolormesh(ce_test_CNNLSTM_Spatial_tw, norm=norm_ce_tw, cmap='jet', shading='auto')
    axs[2, 3].xaxis.set_ticks([])
    axs[2, 3].yaxis.set_ticks([])
    # ConvLSTMStacking
    axs[2, 4].pcolormesh(ce_test_ConvLSTMStacking_Spatial_tw, norm=norm_ce_tw, cmap='jet', shading='auto')
    axs[2, 4].xaxis.set_ticks([])
    axs[2, 4].yaxis.set_ticks([])
    fig.colorbar(c, ax=axs[2, :])

    # MLP
    d = axs[3, 0].pcolormesh(ce_test_MLP_Spatial_dh, norm=norm_ce_dh, cmap='jet', shading='auto')
    axs[3, 0].xaxis.set_ticks([])
    axs[3, 0].yaxis.set_ticks([])
    # axs[1, 0].set_title('MLP', fontsize=16, weight='bold', loc='center')
    axs[3, 0].set_ylabel('CE \n for East China Sea', fontsize=16, weight='bold')
    # LSTM
    axs[3, 1].pcolormesh(ce_test_LSTM_Spatial_dh, norm=norm_ce_dh, cmap='jet', shading='auto')
    axs[3, 1].xaxis.set_ticks([])
    axs[3, 1].yaxis.set_ticks([])
    # axs[1, 0].set_title('MLP', fontsize=16, weight='bold', loc='center')
    # axs[1, 1].set_ylabel('CE', fontsize=16, weight='bold')
    # CNN
    axs[3, 2].pcolormesh(ce_test_CNN_Spatial_dh, norm=norm_ce_dh, cmap='jet', shading='auto')
    axs[3, 2].xaxis.set_ticks([])
    axs[3, 2].yaxis.set_ticks([])
    # CNNLSTM
    axs[3, 3].pcolormesh(ce_test_CNNLSTM_Spatial_dh, norm=norm_ce_dh, cmap='jet', shading='auto')
    axs[3, 3].xaxis.set_ticks([])
    axs[3, 3].yaxis.set_ticks([])
    # ConvLSTMStacking
    axs[3, 4].pcolormesh(ce_test_ConvLSTMStacking_Spatial_dh, norm=norm_ce_dh, cmap='jet', shading='auto')
    axs[3, 4].xaxis.set_ticks([])
    axs[3, 4].yaxis.set_ticks([])
    fig.colorbar(d, ax=axs[3, :])

    plt.savefig('figure4.jpg', dpi=300)
    plt.show()
    print('OK')

if __name__ == '__main__':
    MeticsSpatialDistributionShow3(lead_time=1)