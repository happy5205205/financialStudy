# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:55:20 2019

@author: peng_zhang
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
import re
import datetime
import time
import numpy as np
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

def CareerYear(x):
    # 对工作年限进行修改
    x = str(x)
    if x.find('n/a'):
        return -1
    elif x.find('10+'):
        return 11
    elif x.find('< 1'):
        return 0.5
    else:
        c = re.sub(r'\D', "", x)
        return c


def DescExisting(x):
    # 将desc变量转换成有记录和无记录两种
    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConverDataStr(x):
    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    # time.mktime 不能读取1970年之前的日期
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1', '%Y-%m')))
    else:
        yr = int(x[4:6])
        if yr < 17:
            yr = 2000 + yr
        else:
            yr = 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr, mth, 1)


# 处理缺失值
def MakeupMissing(x):
    if np.isnan(x):
        return -1
    else:
        return x


def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate, earlyDate)
        yr = gap.years
        mth = gap.months
        return yr*12+mth
    else:
        return 0


def BinBadRate(df, col, target, grantRateIndicator=0):
    """
        判断每个类别同时包含好坏样本
        :param df: 需要计算好坏比率的数据集
        :param col: 需要计算好坏比率的特征
        :param target: 好坏标签
        :param grantRateIndicator: 1返回总体的坏样本率，0不返回
        :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total':total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad':bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator == 0:
        return (dicts, regroup)
    else:
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        overallRate = B * 1.0 / N
        return (dicts, regroup, overallRate)


def MergeBad0(df, col, target, direction='bad'):
    """
     每个组都有好坏样本
     :param df: 包含检验0％或者100%坏样本率
     :param col: 分箱后的变量或者类别型变量。检验其中是否有一组或者多组没有坏样本或者没有好样本。如果是，则需要进行合并
     :param target: 目标变量，0、1表示好、坏
     :return: 合并方案，使得每个组里同时包含好坏样本
    """
    regroup = BinBadRate(df=df, col=col, target=target)[1]
    if direction == 'bad':
        # 如果是合并坏样本率为0的组，则跟最小的坏样本率为非0的组进行合并
        regroup = regroup.sort_values(by='bad_rate')
    else:
        # 如果是合并好样本率为0的组，则跟最小的好样本率为非0的组进行合并
        regroup = regroup.sort_values(by='bad_rate', ascending=False)
    regroup.index = range(regroup.shape[0])
    col_regroup = [[i] for i in regroup[col]]
    del_index = []
    for i in range(regroup.shape[0] - 1):
        col_regroup[i+1] = col_regroup[i] + col_regroup[i+1]
        del_index.append(i)
        if direction == 'bad':
            if regroup['bad_rate'][i+1] > 0:
                break
        else:
            if regroup['bad_rate'][i+1] < 1:
                break
    col_regroup2= [col_regroup[i] for i in range(len(col_regroup)) if i not in del_index]
    newGroup = {}
    for i in range(len(col_regroup2)):
        for g2 in col_regroup2[i]:
            newGroup[g2] = 'Bin' + str(i)
    return newGroup


def main():
    data_path = './data'
    outPath = './result/'
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    allData = pd.read_csv(os.path.join(data_path, 'application.csv'))
    allData['term'] = allData['term'].apply(lambda x: int(x.replace('months', '')))
    # target 处理，loan_status标签中Fully Paid是正常客户  Charged Off是违约客户
    allData['target'] = allData['loan_status'].apply(lambda x: int(x == 'Charged Off'))
    '''
    由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
    '''
    allData['term'] = allData[allData['term'] == 36]

    trainData, testData = train_test_split(allData, test_size=1/4)

    # 固话变量
    trainDataFile = open(outPath+'trainData.pkl', 'w')
    pickle.dump(trainData, trainDataFile)
    trainDataFile.close()

    testDataFile = open(outPath+'testDataFile.pkl', 'w')
    pickle.dump(testData, testDataFile)
    testDataFile.close()

    '''
        第一步数据预处理
        1、数据清洗
        2、格式转换
        3、缺失值处理
        
    '''

    # 将带%的百分比变为浮点数
    trainData['int_rate_clean'] = trainData['int_rate'].apply(lambda x: float(x.replace('%', ''))/100)

    # 将工作年限进行转换
    trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)

    # 将desc的缺失作为一种状态，非缺失作为另一种状态
    trainData['desc_clean'] = trainData['desc'].map(DescExisting)

    # 处理日期 earliest_cr_line标签的格式不统一，需要统一格式且转换成python格式
    trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(ConverDataStr)

    trainData['app_data_clean'] = trainData['issue_d'].map(ConverDataStr)

    #  对缺失值进行处理
    trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(MakeupMissing)
    trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(MakeupMissing)
    trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(MakeupMissing)

    '''
        第二步：变量的衍生
    '''

    # 考虑申请额度于收入的占比
    trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt/x.annual_inc, axis=1)
    # 考虑earliest_cr_line到申请日期的跨度，以月份记
    trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean, x.app_data_clean), axis=1)

    '''
        第三步：分箱，采用ChiMerge，要求分箱完成之后
            1、不超过5箱
            2、Bad Rate单调
            3、每箱同时包括好坏样本
            4、特殊值如-1， 单独成一箱
        连续型变量可以直接分箱
        类别型变量
            1、当取值较多时，先用bad rate编码，再用连续分箱的方法进行分箱
            2、当取值较少时
                1、如果每种类别同时包含好坏样本，无需分箱
                2、如果有类别只包含好、坏样本的一种，需要合并
    '''

    # 连续型变量
    num_features = ['int_rate_clean', 'emp_length_clean', 'annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app',
                    'inq_last_6mths', 'mths_since_last_record_clean', 'mths_since_last_delinq_clean', 'open_acc',
                    'pub_rec', 'total_acc', 'limit_income', 'earliest_cr_to_app']

    # 类别型变量
    cat_features = ['home_ownership', 'verification_status', 'desc_clean', 'purpose', 'zip_code', 'addr_state',
                    'pub_rec_bankruptcies_clean']
    more_value_feature = []  # 存放变量取值大于5
    less_value_feature = []  # 存放变量取值少于5

    for col in cat_features:
        valuesCounts = len(set(trainData[col]))
        if valuesCounts > 5:
            more_value_feature.append(col)  # 取值超过5的变量，需要bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_feature.append(col)  # 如果每种类别同时包含好坏样本，无需分箱, 如果有类别只包含好坏样本的一种，需要合并

    # 取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
    merge_bin_dict = {} # 存放需要合并的变量，以及合并方法
    var_bin_liat = {} # 由于某个取值没有好或者坏样本而需要合并的变量

    for col in less_value_feature:
        binBadRate = BinBadRate(df=trainData, col=col, target='target')[0]
        print '{}的取值根据标签分组不同属性的坏样本比例为{}'.format(col,binBadRate)
        if min(binBadRate.values()) == 0:
            print '{}标签中存在坏样本比例为0，需要合并'.format(col)
            combine_bin = MergeBad0(df=trainData, col=col, target='target')
            print combine_bin
        else:
            pass



if __name__ == '__main__':
    main()

