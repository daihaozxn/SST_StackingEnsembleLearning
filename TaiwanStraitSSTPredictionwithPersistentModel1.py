import ipykernel, h5py
from sklearn.metrics import mean_squared_error
from numpy import mean, sum

lead_time = 7
n_steps = 4
# f = h5py.File('donghai_raw_train.h5', 'r')
# raw_train = f.get('donghai_raw_train')[:]
f = h5py.File('donghai_raw_test.h5', 'r')
raw_test = f.get('donghai_raw_test')[:]

yhat = raw_test[(n_steps-1):-lead_time,:]
y = raw_test[(n_steps+lead_time-1):,:]

yhat = yhat.reshape(len(yhat), -1)
y = y.reshape(len(y), -1)

rmse_persistent = mean_squared_error(y, yhat, squared=False)
print('rmse_persistent: %.6f ℃' % rmse_persistent)

ce_persistent = 1 - (sum((y - yhat) ** 2)) / (sum((y - mean(y)) ** 2))
print('ce_persistent: %.6f' % ce_persistent)