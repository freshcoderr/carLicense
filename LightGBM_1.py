'''
Without any action to data, ref:https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.4bcac0948RI9a9&raceId=231641&postsId=3809
'''
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

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1] # n_var: number of cols(attribute)
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

filename = 'data/'
train = pd.read_table(filename + 'train_20171215.txt', engine='python')
test_A = pd.read_table(filename + 'test_A_20171225.txt', engine='python')
sample_A = pd.read_table(filename + 'sample_A_20171225.txt', engine='python', header=None)
'''
remove the label'brand'
'''
train = train.groupby(['date', 'day_of_week'], as_index=False).cnt.sum()
time_cnt = list(train['cnt'].values)
#time_cnt.append(0)
time2sup = series_to_supervised(data=time_cnt, n_in=276, dropnan =True)

index_split = len(time_cnt)*0.732
x_train = time2sup[time2sup.index<index_split]
x_test = time2sup[time2sup.index>index_split]

y_train = x_train.pop('var1(t)')
y_test = x_test.pop('var1(t)')

'''
print(time2sup.shape)
time2sup_target = time2sup['var1(t)'].values
time2sup_data =time2sup.drop(['var1(t)'], axis=1).values
print(time2sup.shape)

cv = cross_validation.ShuffleSplit(len(time2sup_target), n_iter=5, test_size=0.3,random_state=0)
for train,test in cv:
    train_target = time2sup_target[train]
    train_data = time2sup_data[train]
    test_target = time2sup_target[test]
    test_data = time2sup_data[test]

gbm0 = lgb.LGBMRegressor(
	objective='regression',
	num_leaves=64,
	learning_rate=0.05,
	n_estimator=10000)
print(test_data.shape)
print(x_test.shape)

print("split by random")
gbm0.fit(train_data, train_target,eval_set=[(test_data,test_target)],eval_metric='mse',early_stopping_rounds=15)
result_random = gbm0.predict(test_data)
print(mean_squared_error(result_random, test_target))
'''
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
    time2sup = series_to_supervised(data=time_cnt, n_in=276, dropnan =True)
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
plt.show()


