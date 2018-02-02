import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pandas import DataFrame
from pandas import concat
import lightgbm as lgb
from scipy import stats
from scipy.stats import norm, skew

filename = 'data/'
train = pd.read_table(filename + 'train_20171215.txt', engine='python')
test_A = pd.read_table(filename + 'test_A_20171225.txt', engine='python')
sample_A = pd.read_table(filename + 'sample_A_20171225.txt', engine='python', header=None)


group_day = train.groupby(['date', 'day_of_week'], as_index=False).cnt.sum()
mean_week = group_day.groupby('day_of_week', as_index=False).cnt.mean()

print mean_week
group_week = group_day.groupby('day_of_week')
plot_week_num = 331

i = 0
Monday_data = pd.DataFrame()
Tuesday_data = pd.DataFrame()
Wednesday_data = pd.DataFrame()
Thursday_data = pd.DataFrame()
Friday_data = pd.DataFrame()
Saturday_data = pd.DataFrame()
Sunday_data = pd.DataFrame()
#fig1 = plt.figure(1)
for g in zip(group_week):
    if i == 0:
        plt.subplot(3,3,1)
        Monday_data = g[0][1]
    elif i == 1:
        plt.subplot(3, 3, 2)
        Tuesday_data = g[0][1]
    elif i == 2:
        plt.subplot(3, 3, 3)
        Wednesday_data = g[0][1]
    elif i == 3:
        plt.subplot(3, 3, 4)
        Thursday_data = g[0][1]
    elif i == 4:
        plt.subplot(3, 3, 5)
        Friday_data = g[0][1]
    elif i ==5:
        plt.subplot(3, 3, 6)
        Saturday_data = g[0][1]
    elif i ==6:
        plt.subplot(3, 3, 7)
        Sunday_data = g[0][1]
    i += 1


#    print len(g[0][1].date)
    #plt.plot(g[0][1].date, g[0][1].cnt.values)

#fig1.show()
'''
# FFT Transfer
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
fft_data = Monday_data
x_Mon_fft = fft_data.date.values
y_Mon_fft = fft_data.cnt.values
yy = fft(y_Mon_fft)

yf=abs(fft(y_Mon_fft))
yf1=abs(fft(y_Mon_fft))/len(x_Mon_fft)
yf2 = yf1[range(int(len(x_Mon_fft)/2))]

xf = np.arange(len(y_Mon_fft))
xf1 = xf
xf2 = xf[range(int(len(x_Mon_fft)/2))]
plt.figure(2)
plt.subplot(221)
plt.plot(x_Mon_fft[:],y_Mon_fft[:])
plt.title('Original wave')

plt.subplot(222)
plt.plot(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')
plt.subplot(223)
plt.plot(xf1,yf1,'g')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

plt.subplot(224)
plt.plot(xf2,yf2,'b')
plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
'''

from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



plt.show()
