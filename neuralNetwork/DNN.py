# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:55:20 2019

@author: peng_zhang
"""

import pandas as pd
import  tensorflow as tf
import re
import datetime
import time
from dateutil.relativedelta import relativedelta
import os
from sklearn.model_selection import train_test_split

#normalize the features using max-min to convert the values into [0,1] interval
def MaxMinNorm(df, col):
    ma, mi = max(df[col]),min(df[col])
    rangeVal = ma - mi
    if rangeVal == 0:
        print(col)
    df[col] = df[col].apply(lambda x: (x-mi)*1.0/rangeVal)


def CareerYear(x):
    x = str(x)
    if not x == x:
        return -1
    elif x.find('+10')>-1:
        return 11
    elif x.find('< 1') > -1:
        return 0
    else:
        return int(re.sub('\D', '', x))

def DescExisting(x):
    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConvertDateStr(x):
    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    if str(x) == 'nan':
        return datetime.datetime.fromisoformat(time.mktime(time.strptime('9900-1', '%Y-%m')))
    else:
        yr = int(x[4:6])
        if yr <= 17:
            yr = 2000+yr
        else:
            yr = 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr, mth, 1)


def MonthGap(earlyDate, lataDate):
    if lataDate > earlyDate:
        gap =relativedelta(lataDate, earlyDate)
        yr = gap.years
        mth = gap.months
        return yr*12 +mth
    else:
        return 0


def MakeupMissing(x):
    if not x==x:
        return -1
    else:
        return x

"""
    第一步：数据准备
"""
data = './data/'

allData = pd.read_csv(os.path.join(data, 'application.csv'), header=0, encoding='latin1')
allData['term'] = allData['term'].apply(lambda x: int(x.replace('months', '')))

# 处理标签：Fully Paid是正常用户；Charged Off是违约用户
allData['target'] = allData['loan_status'].apply(lambda x: int(x == 'Charged Off'))

'''
由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
'''
allData1 = allData.loc[allData.term == 36]
trainData, testData = train_test_split(allData1, test_size=0.4)




