# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:04:22 2019

@author: peng_zhang
"""

import pandas as pd
import numpy as np


def DelqFeatures(event, window, type):
    """

        :param event:
        :param window:
        :param type:
        :return:
    """
    current = 12
    start = 12 - window + 1
    delq1 = [event[a] for a in ['Delq1_'+str(t) for t in range(current, start - 1, -1)]]
    delq2 = [event[a] for a in ['Delq2_' + str(t) for t in range(current, start - 1, -1)]]
    delq3 = [event[a] for a in ['Delq3_' + str(t) for t in range(current, start - 1, -1)]]

    if type == 'max delq':
        if max(delq3) == 1:
            return 3
        elif max(delq2) == 1:
            return 2
        elif max(delq1) == 1:
            return 1
        else:
            return 0
    if type in ['M0 times', 'M1 times', 'M2 times']:
        if type.find('M0') > -1:
            return sum(delq1)
        elif type.find('M1') > -1:
            return sum(delq2)
        else:
            return sum(delq3)


def UrateFeatures(event, window, type):
    """

        :param event:
        :param window:
        :param type:
        :return:
    """
    current = 12
    start = 12 - window + 1
    monthlySpend = [event[a] for a in ['Spend_' + str(t) for t in range(current, start - 1, -1)]]
    limit = event['Loan_Amount']
    monthlyUrate = [x / limit for x in monthlySpend]
    if type == 'mean utilization rate':
        return np.mean(monthlyUrate)
    if type == 'max utilization rate':
        return max(monthlyUrate)
    if type == 'increase utilization rate':
        currentUrate = monthlyUrate[0:-1]
        previousUrate = monthlyUrate[1:]
        compareUrate = [int(x[0] > x[1]) for x in zip(currentUrate, previousUrate)]
        return sum(compareUrate)


def PaymentFeatures(event, window, type):
    """

        :param event:
        :param window:
        :param type:
        :return:
    """
    current = 12
    start = 12 - window + 1
    currentPayment = [event[a] for a in ['Payment_' + str(t) for t in range(current, start - 1, -1)]]
    previousOS = [event[a] for a in ['OS_' + str(t) for t in range(current - 1, start - 2, -1)]]
    monthlyPayRatio = []
    for Pay_OS in zip(currentPayment, previousOS):
        if Pay_OS[1] > 0:
            payRatio = Pay_OS[0]*1.0 / Pay_OS[1]
            monthlyPayRatio.append(payRatio)
        else:
            monthlyPayRatio.append(1)
    if type == 'min payment ratio':
        return min(monthlyPayRatio)
    if type == 'max payment ratio':
        return max(monthlyPayRatio)
    if type == 'mean payment ratio':
        total_payment = sum(currentPayment)
        total_OS = sum(previousOS)
        if total_OS > 0:
            return total_payment / total_OS
        else:
            return 1


def BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby(col)[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby(col)[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, how='left', right_index=True, left_index=True)
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total, axis=1)
    dicts = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dicts, regroup)
    else:
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        overallRate = B / N
        return (dicts, regroup, overallRate)

def BadRateMonotone(df, sortByVar, target,special_attribute = []):
    """
        判断某变量的坏样本率是否单调
        :param df: 包含检验坏样本率的变量，和目标变量
        :param sortByVar: 需要检验坏样本率的变量
        :param target: 目标变量，0、1表示好、坏
        :param special_attribute: 不参与检验的特殊值
        :return: 坏样本率单调与否
    """
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1]/x[0] for x in combined]
    badRateNotMonotone = [badRate[i] < badRate[i+1] and badRate[i] < badRate[i-1] or
                          badRate[i] > badRate[i+1] and badRate[i] > badRate[i-1]
                          for i in range(1, len(badRate) - 1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


def CalcWOE(df, col, target):
    """
        计算WOE和IV值
        :param df: 包含需要计算WOE的变量和目标变量
        :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
        :param target: 目标变量，0、1表示好、坏
        :return: 返回WOE和IV
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].apply(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].apply(lambda x: x*1.0/G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt), axis=1)
    IV_value = sum(IV)
    return {"WOE": WOE_dict, 'IV': IV_value}

#################################
#   1, 读取数据，衍生初始变量   #
#################################
folderOfData = 'D:/financialStudy/Bcard/data/'
trainData = pd.read_csv(folderOfData+'trainData.csv', header=0)

allFeatures = []
'''
逾期类型的特征在行为评分卡（预测违约行为）中，一般是非常显著的变量。
通过设定时间窗口，可以衍生以下类型的逾期变量：
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1, 3, 6, 12]:

    # 1，过去t时间窗口内的最大逾期状态
    allFeatures.append('maxDelqL'+str(t)+'M')
    trainData['maxDelqL'+str(t)+'M'] = trainData.apply(lambda x: DelqFeatures(x, t, 'max delq'), axis=1)

    # 2，过去t时间窗口内的，M0,M1,M2的次数
    allFeatures.append('M0FreqL' + str(t) + "M")
    trainData['M0FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M0 times'), axis=1)

    allFeatures.append('M1FreqL' + str(t) + "M")
    trainData['M1FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M1 times'), axis=1)

    allFeatures.append('M2FreqL' + str(t) + "M")
    trainData['M2FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M2 times'), axis=1)

'''
额度使用率类型特征在行为评分卡模型中，通常是与违约高度相关的
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1, 3, 6, 12]:
    # 1，过去t时间窗口内的最大月额度使用率
    allFeatures.append('maxUrateL'+str(t)+'M')
    trainData['maxUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'max utilization rate'),
                                                            axis=1)

    # 2，过去t时间窗口内的平均月额度使用率
    allFeatures.append('avgUrateL' + str(t) + "M")
    trainData['avgUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'mean utilization rate'),
                                                            axis=1)
    # 3，过去t时间窗口内，月额度使用率增加的月份。该变量要求t>1
    if t > 1:
        allFeatures.append('increaseUrateL' + str(t) + "M")
        trainData['increaseUrateL' + str(t) + "M"] = trainData.apply(
                                                    lambda x: UrateFeatures(x, t, 'increase utilization rate'), axis=1)

'''
还款类型特征也是行为评分卡模型中常用的特征
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1, 3, 6, 12]:
    # 1，过去t时间窗口内的最大月还款率
    allFeatures.append('maxPayL' + str(t) + "M")
    trainData['maxPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'max payment ratio'), axis=1)

    # 2，过去t时间窗口内的最小月还款率
    allFeatures.append('minPayL' + str(t) + "M")
    trainData['minPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'min payment ratio'), axis=1)

    # 3，过去t时间窗口内的平均月还款率
    allFeatures.append('avgPayL' + str(t) + "M")
    trainData['avgPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'mean payment ratio'), axis=1)

'''
类别型变量：过去t时间内最大的逾期状态
需要检查与bad的相关度
'''
trainData.groupby(['maxDelqL1M'])['label'].mean()
trainData.groupby(['maxDelqL3M'])['label'].mean()
trainData.groupby(['maxDelqL6M'])['label'].mean()
trainData.groupby(['maxDelqL12M'])['label'].mean()

for x in allFeatures:
    for y in allFeatures:
        if x != y:
            print(x, '   ', y, '    ', np.corrcoef(trainData[x], trainData[y])[0, 1])

############################
#   2, 分箱，计算WOE并编码   #
############################

'''
对类别型变量的分箱和WOE计算
可以通过计算取值个数的方式判断是否是类别型变量
'''
WOE_IV_dict = {}
categoricalFeatures = [var for var in allFeatures if len(set(trainData[var])) <= 5]  # 类别型变量
numericalFeatures = [var for var in allFeatures if len(set(trainData[var])) > 5]  # 数字型变量

# 检查类别型变量bad rate在箱中的单调性
not_monotone = [var for var in categoricalFeatures if not BadRateMonotone(trainData, var, 'label')]
# ['M1FreqL3M', 'M2FreqL3M', 'maxDelqL12M']

trainData.groupby('M1FreqL3M')['label'].mean()
trainData.groupby('M1FreqL3M')['label'].count()

trainData['M1FreqL3M_Bin'] = trainData['M1FreqL3M'].apply(lambda x: int(x >= 1))

WOE_IV_dict['M1FreqL3M_Bin'] = CalcWOE(trainData, 'M1FreqL3M_Bin', 'label')


trainData.groupby('M2FreqL3M')['label'].mean()
trainData.groupby('M2FreqL3M')['label'].count()

trainData['M2FreqL3M_Bin'] = trainData['M2FreqL3M'].apply(lambda x: int(x >= 1))

WOE_IV_dict['M2FreqL3M_Bin'] = CalcWOE(trainData, 'M2FreqL3M_Bin', 'label')

trainData.groupby('maxDelqL12M')['label'].mean()
trainData.groupby('maxDelqL12M')['label'].count()

'''
对其他单调的类别型变量，检查是否有一箱的占比低于5%。 如果有，将该变量进行合并
'''