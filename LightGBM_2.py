'''
split by week
'''
import Time_convert as Tc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pandas import DataFrame
from pandas import concat
import lightgbm as lgb
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import ShuffleSplit
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

filename = 'data/'
train = pd.read_table(filename + 'train_20171215.txt', engine='python')
test_A = pd.read_table(filename + 'test_A_20171225.txt', engine='python')
sample_A = pd.read_table(filename + 'sample_A_20171225.txt', engine='python', header=None)
'''
remove the label'brand'
'''
group_day = train.groupby(['date', 'day_of_week'], as_index=False).cnt.sum()
mean_week = group_day.groupby('day_of_week', as_index=False).cnt.mean()
print mean_week
group_week = group_day.groupby('day_of_week')

i = 0
Monday_data = pd.DataFrame()
Tuesday_data = pd.DataFrame()
Wednesday_data = pd.DataFrame()
Thursday_data = pd.DataFrame()
Friday_data = pd.DataFrame()
Saturday_data = pd.DataFrame()
Sunday_data = pd.DataFrame()

for g in zip(group_week):
    if i == 0:
        Monday_data = g[0][1]
    elif i == 1:
        Tuesday_data = g[0][1]
    elif i == 2:
        Wednesday_data = g[0][1]
    elif i == 3:
        Thursday_data = g[0][1]
    elif i == 4:
        Friday_data = g[0][1]
    elif i ==5:
        Saturday_data = g[0][1]
    elif i ==6:
        Sunday_data = g[0][1]
    i += 1

# decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
Monday_series = Monday_data.drop(['day_of_week'], axis=1).cnt
Monday_series.index = pd.to_datetime(Monday_series.index, unit='D')
log_Mon = np.log(Monday_series)
log_Mon = Monday_series
decomposition_Mon = seasonal_decompose(log_Mon,  freq=30, model='multiplicative')
Trend_Mon= decomposition_Mon.trend
seasonal_Mon = decomposition_Mon.seasonal
residual_Mon = decomposition_Mon.resid
recover = []
for i in range(len(log_Mon)):
    if math.isnan(Trend_Mon[i]):
        Trend_tmp = 1
    else:
        Trend_tmp = Trend_Mon[i]
    if math.isnan(seasonal_Mon[i]):
        seasonal_tmp = 1
    else:
        seasonal_tmp = seasonal_Mon[i]
    if math.isnan(residual_Mon[i]):
        residual_tmp = 1
    else:
        residual_tmp = residual_Mon[i]
    recover.append(Trend_tmp*seasonal_tmp*residual_tmp)


# plt.subplot(511)
# plt.plot(log_Mon)
# plt.subplot(512)
# plt.plot(Trend_Mon)
# plt.subplot(513)
# plt.plot(seasonal_Mon)
# plt.subplot(514)
# plt.plot(residual_Mon)
# plt.subplot(515)
# plt.plot(recover)

import myutils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
dataset = Trend_Mon.to_frame(name=None).dropna().values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset)*0.67)
test_size = len(dataset) - train_size
train, test = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

for i in range(20):
    print ("ROLL %d" %(i))
    look_back = 1
    trainX, trainY = myutils.create_dataset(train, look_back)
    testX, testY = myutils.create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)


    # testX.append(testPredict[-1])
    # trainX.append(testX[i])
    # trainY.append(test[i+1])
    tmp = np.empty_like(train)
    tmp[0:len(train)-1, :] = train[1:len(train), :]
    tmp[-1, :] = test[0, :]
    train = tmp
    tmp2 = np.empty_like(test)
    tmp2[0:len(test)-1, :] = test[1:len(test), :]
    tmp2[-1, :] = testPredict[-1, :]
    test = tmp2


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()




'''
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
Trend_Mon = Trend_Mon.dropna()

df_train_target = Trend_Mon.asobject
df_train_data = Monday_data.drop(['day_of_week','cnt'], axis=1).values
df_train_data = df_train_data[15:145]
xx = []
for i in range(35):
    xx.append(df_train_data[-1] + 7*(i + 1))
df_test_x = np.array(xx)
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05,
                              n_restarts_optimizer=10)
X = df_train_data
y = df_train_target
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(df_test_x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(df_test_x, y_pred, 'b-', label=u'Prediction')
plt.plot(X, y, 'r', label=u'true')
'''
'''
train = Monday_data
time_cnt = list(train['cnt'].values)
#time_cnt.append(0)
time2sup = Tc.series_to_supervised(data=time_cnt, n_in=276, dropnan =True)

index_split = len(time_cnt)*0.732
x_train = time2sup[time2sup.index<index_split]
x_test = time2sup[time2sup.index>index_split]

y_train = x_train.pop('var1(t)')
y_test = x_test.pop('var1(t)')


print("==========================================")
gbm1 = lgb.LGBMRegressor(
	objective='regression',
	num_leaves=64,
	learning_rate=0.05,
	n_estimator=10000)

gbm1.fit(x_train.values,y_train,eval_set=[(x_test.values,y_test)],eval_metric='mse',early_stopping_rounds=15)
print("split at 0.7 and result:")
result_convert = gbm1.predict(x_test.values)
y_test = list(y_test)
result_convert = list(result_convert)

print ("%d vs %d" %(result_convert[-1], y_test[-1]))
#print(mean_squared_error(result_convert, y_test))

lens_A = len(test_A)
for i in range(lens_A):
    time_cnt.append(0)
    time2sup = Tc.series_to_supervised(data=time_cnt, n_in=276, dropnan =True)
    index_split = len(time_cnt)*0.732
    #x_train = time2sup[time2sup.index<index_split]
    x_test = time2sup[time2sup.index>index_split]
    #y_train = x_train.pop('var1(t)')
    y_test = x_test.pop('var1(t)')
    result_convert = list(gbm1.predict(x_test.values))
    time_cnt[-1] = int(result_convert[-1])
    test_A.loc[i,'day_of_week'] = int(result_convert[-1])


test_A[['date','day_of_week']].to_csv('result.txt',index=False,header=False,sep='\t')



from sklearn.metrics import mean_squared_error
line1 = plt.plot(range(len(x_test)),test_A['day_of_week'].values,label=u'predict')
line2 = plt.plot(range(len(y_test)),y_test.values,label=u'true')
plt.legend()


'''
#plt.show()
