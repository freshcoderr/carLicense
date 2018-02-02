import pandas as pd
import matplotlib.pyplot as plt
filename = "/home/raphael/tianchi/carlicense/result.txt"
result = pd.read_table(filename, engine='python', names=['date', 'cnt'])
filename = 'carLicense/data/'
train = pd.read_table(filename + 'train_20171215.txt', engine='python')
train = train.groupby(['date', 'day_of_week'], as_index=False).cnt.sum()
plt.plot(result['date'].values, result['cnt'].values, label='predict')
plt.plot(train['date'].values, train['cnt'].values, label='true')
plt.legend
plt.show()
