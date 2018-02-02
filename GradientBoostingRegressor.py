'''
Gradient Tree Boosting GBRT
'''
import pandas as pd
import numpy as np

train = pd.read_table('data/train_20171215.txt',engine='python')
train.describe()

actions1 = train.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})

from sklearn import cross_validation
from sklearn import svm
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

df_train_target = actions1['count1'].values
df_train_data = actions1.drop(['count1'],axis = 1).values

cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=5, test_size=0.2,random_state=0)

print "GradientBoostingRegressor"  
'''
GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
'''  
for train, test in cv:    
    gbdt = GradientBoostingRegressor().fit(df_train_data[train], df_train_target[train])
    result1 = gbdt.predict(df_train_data[test])
    print(mean_squared_error(result1,df_train_target[test]))
    print '......'


#test_A[['date','result']].to_csv('result.txt',index=False,header=False,sep='\t')
