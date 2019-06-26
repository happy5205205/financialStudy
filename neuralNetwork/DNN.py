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
import warnings
warnings.filterwarnings('ignore')

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
        c = re.sub(r'\D', "", x)
        return c

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

'''
    第一步：数据准备
'''

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

'''
    第二步：试据预处理
'''
# 将带%的百分比变为浮点数
trainData['int_rate_clean'] = trainData['int_rate'].apply(lambda x: float(x.replace('%', ''))/100)
# 将工作年限进行转换，否则影响排序
trainData['emp_length_clean'] = trainData['emp_length'].apply(lambda x: CareerYear(x))
# 将desc的缺失作为一种状态，非缺失作为另一种状态
trainData['desc_clean'] = trainData['desc'].apply(lambda x: DescExisting(x=x))

# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
trainData['app_date_clean'] = trainData['issue_d'].apply(lambda x: ConvertDateStr(x=x))
trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].apply(lambda x: ConvertDateStr(x=x))

# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x: MakeupMissing(x))
trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x: MakeupMissing(x))
trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x: MakeupMissing(x))

'''
    第三步：变量衍生
'''
# 考虑申请额度和收入的占比
trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt/x.annual_inc, axis=1)
# 考虑earliest_cr_line到申请日期的跨度，以月份记
trainData['earliest_cr_to_app'] = trainData.apply(lambda x:MonthGap(x.earliest_cr_line_clean, x.app_date_clean), axis=1)

'''
    对于类别型变量，需要onehot（独热）编码
'''
# 数字型变量
num_features = ['int_rate_clean', 'emp_length_clean', 'annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app', 'inq_last_6mths',
                'mths_since_last_record_clean', 'mths_since_last_delinq_clean', 'open_acc', 'pub_rec', 'total_acc', 'limit_income', 'earliest_cr_to_app']
# 类别型变量
cat_features = ['home_ownership', 'verification_status', 'desc_clean', 'purpose', 'zip_code', 'addr_state', 'pub_rec_bankruptcies_clean']

from sklearn.feature_extraction import DictVectorizer

# 将类别型特征进行独热编码 DictVectorizer这个函数可以
v = DictVectorizer(sparse=False)
X1 = v.fit_transform(trainData[cat_features].to_dict('records'))
feature_name = v.get_feature_names()
# 将独热编码和数值型变量放在一起进行模型训练
import numpy as np
X2 = np.matrix(trainData[num_features])
X = np.hstack([X1, X2])
Y = trainData['target']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# numnber of input layer nodes: dimension =
# number of hidden layer & number of nodes in them: hidden_units
# full link or not: droput. dropout = 1 means full link
# activation function: activation_fn. By default it is relu
# learning rate:


# Example: select the best number of units in the 1-layer hidden layer
# model_dir = path can make the next iteration starting from last termination
# define the DNN with 1 hidden layer

no_hidden_units_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column('', dimension = x_train.shape[1])]
from tensorflow.contrib.learn.python.learn.estimators import SKCompat
from sklearn.metrics import roc_auc_score
for no_hidden_units in range(10, 51, 10):
    print("the current choise of hidden units number is {}".format(no_hidden_units))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[no_hidden_units, no_hidden_units+10],n_classes=2,dropout=0.5)
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256, steps=1000)
    clf_pred_proba = clf._estimator.predict_proba(x_test)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_test, pred_proba)
    no_hidden_units_selection[no_hidden_units] = auc_score
best_hidden_units = max(no_hidden_units_selection.items(), key=lambda x: x[1])[0]



# for dropout_prob in np.linspace(0,0.99,100):
#     print("the current choise of drop out rate is {}".format(dropout_prob))
