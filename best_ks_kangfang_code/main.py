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


def BadRateEncoding(df, col, target):
    """
    变量属性大于5的变量通过坏样本率进行编码
    :param df: dataframe containing feature and target
    :param col: the feature that needs to be encoded with bad rate, usually categorical type
    :param target: good/bad indicator
    :return: the assigned bad rate to encode the categorical feature
    """
    regroup = BinBadRate(df=df, col=col, target=target, grantRateIndicator=0)[1]
    br_dict = regroup[[col, 'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x:br_dict[x])
    return {'encoding': badRateEnconding, 'bad_rate': br_dict}


'''
    分界线，下列函数全用于卡方分箱
'''


def SplitData(df, col, numOfSplit, special_attribute=[]):

    """
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    :return: 在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    """
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = N / numOfSplit
    splitPointIndex = [i*n for i in range(1, numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint


def AssignGroup(x, bin):
    """
    :param x: 某个变量的某个取值
    :param bin: 上述变量的分箱结果
    :return: x在分箱结果下的映射
    """
    N = len(bin)
    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]


def Chi2(df, total_col, bad_col):
    """
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :return: 卡方值
    """
    df2 = df.copy()
    # 求出df中总体的坏样本率和好样本率
    badRate = sum(df2[bad_col])*1.0 / sum(df2[total_col])
    df2['good'] = df2.apply(lambda x: x[total_col] - x[bad_col], axis=1)
    # df2['good'] = df2.apply(lambda x: x.total_col-x.bad_col, axis=1)
    goodRate = sum(df2['good'])*1.0 / sum(df2[total_col])
    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df[total_col].apply(lambda x: x*badRate)
    df2['goodExpected'] = df[total_col].apply(lambda x: x*goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined  =zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0]-i[1])**2/(i[0]+0.00001) for i in badCombined]
    goodChi = [(i[0]-i[1])**2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2


def AssignBin(x, cutOffPoints,special_attribute=[]):
    """
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    """
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x <= cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0, numBin-1):
            if cutOffPoints[i] < x < cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)


def ChiMerge(df, col, target, max_interval=5, special_attribute=[], minBinPcnt=0):
    """
    :param df: 包含目标变量与分箱属性的数据框
    :param col: 需要分箱的属性
    :param target: 目标变量，取值0或1
    :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数
    :param special_attribute: 不参与分箱的属性取值
    :param minBinPcnt：最小箱的占比，默认为0
    :return: 分箱结果
    """
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:
        print "The number of original levels for {} is less than or equal to max intervals".format(col)
        return colLevels[:-1]
    else:
        if len(special_attribute) >=1:
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))

        # 步骤一：通过col对数据集进行分组，求出每组的总样本数和坏样本数
        if N_distinct > 100:
            split_x = SplitData(df=df2, col=col, numOfSplit=100)
            print split_x
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df2['temp'] = df2[col]
        # 总体bad rate将被用来计算expected bad count
        (binBadRate, regroup, overallRate) = BinBadRate(df=df2, col='temp', target=target, grantRateIndicator=1)
        # 首先，每个单独的属性值将被分为单独的一组
        # 对属性值进行排序，然后两两组别进行合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]

        #  步骤二建立循环，不断合并最优的相邻两个组别，直到
        #  1，最终分裂出来的分箱数<＝预设的最大分箱数
        #  2，每箱的占比不低于预设值（可选）
        #  3，每箱同时包含好坏样本
        #  如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        split_intervals = max_interval - len(special_attribute)
        while len(groupIntervals) > split_intervals: # 终止条件：当前分箱数==预设分箱数
            # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
            chisqList = []
            for k in range(len(groupIntervals) -1):
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad')
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined + 1]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[best_comnbined + 1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        #  检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
        groupdvalues = df2['temp'].apply(lambda x: AssignBin(x=x, cutOffPoints=cutOffPoints))
        df2['temp_Bin'] = groupdvalues
        (binBadRate, regroup) = BinBadRate(df=df2, col='temp_Bin', target=target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
        while minBadRate == 0 or maxBadRate == 1:
            # 找出全部为好／坏样本的箱
            indexForBad01 = regroup[regroup['bad_rate'].isin([0, 1])].temp_Bin.tolist()
            bin = indexForBad01[0]
            # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
            else:
                # 和前一箱进行合并，并且计算卡方值
                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3= df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df=df3, col='temp_Bin', target='target')
                chisq1 = Chi2(df2b, 'total', 'bad')
                # 和后一箱进行合并，并且计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                chisq2 = Chi2(df2b, 'total', 'bad')
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])

            # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupdvalues = df2['temp'].apply(lambda x: AssignBin(x=x, cutOffPoints=cutOffPoints))
            df2['temp_Bin'] = groupdvalues
            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]

        # 需要检查分箱后的最小占比
        if minBinPcnt > 0:
            groupdvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupdvalues
            valuesCounts = groupdvalues.value_counts().to_frame()
            N = sum(valuesCounts['temp'])
            valuesCounts['pcnt'] = valuesCounts['temp'].apply(lambda x: x*1.0 / N)
            valuesCounts = valuesCounts.sort_index()
            minPcnt = min(valuesCounts['pcnt'])
            while minPcnt < minBinPcnt and len(cutOffPoints) >2:
                # 找出占比最小的箱
                indexForMinPcnt = valuesCounts[valuesCounts['pcnt'] == minPcnt].index.tolist()[0]
                # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
                if indexForMinPcnt == max(valuesCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
                elif indexForMinPcnt == min(valuesCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                else:

                    # 和前一箱进行合并，并且计算卡方值
                    currentIndex = list(valuesCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valuesCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                    chisq1 = Chi2(df2b, 'total', 'bad')

                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valuesCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                    # chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                    chisq2 = Chi2(df2b, 'total', 'bad')
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])
        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints




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
    #  (i)取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
    merge_bin_dict = {}  # 存放需要合并的变量，以及合并方法
    var_bin_list = []  # 由于某个取值没有好或者坏样本而需要合并的变量

    for col in less_value_feature:
        binBadRate = BinBadRate(df=trainData, col=col, target='target')[0]
        print '{}的取值根据标签分组不同属性的坏样本比例为{}'.format(col,binBadRate)
        if min(binBadRate.values()) == 0:
            print '{}标签中存在坏样本比例为0，需要合并'.format(col)
            combine_bin = MergeBad0(df=trainData, col=col, target='target')
            print combine_bin
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            trainData[newVar] = trainData[col].map(combine_bin)
            var_bin_list.append(newVar)

        if max(binBadRate.values()) == 1:
            print '{}标签中存在好样本比例为0，需要合并'.format(col)
            combine_bin = MergeBad0(df=trainData, col=col, target='target')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            trainData[newVar] = trainData[col].map(combine_bin)
            var_bin_list.append(newVar)

    # 保存需要合并的变量，以及合并方法merge_bin_dict
    merge_bin_dict_file = open(outPath + 'merge_bin_dict.pkl', 'w')
    pickle.dump(merge_bin_dict, merge_bin_dict_file)
    merge_bin_dict_file.close()

    # less_value_feature中剔除不需要合并的变量
    less_value_feature = [i for i in less_value_feature if i+'_Bin' not in var_bin_list]

    # （ii）当取值>5时：用bad rate进行编码，放入连续型变量里
    br_encoding_dict = {}  # 记录按照bad rate进行编码的变量，及编码方式
    for col in more_value_feature:
        br_encoding = BadRateEncoding(df=trainData, col=col, target='target')
        print br_encoding
        trainData[col+'_br_encoding'] = br_encoding['encoding']
        br_encoding_dict[col] = br_encoding['bad_rate']
        num_features.append(col+'_br_encoding')

    # 保存需要用坏样本率编码的变量br_encoding_dict
    br_encoding_dict_file = open(outPath+'br_encoding_dict.pkl', 'w')
    pickle.dump(br_encoding_dict, br_encoding_dict_file)
    br_encoding_dict_file.close()

    # （iii）对连续型变量进行分箱，包括（ii）中的变量
    continous_merged_dict = {}
    for col in num_features:
        print '{} is in processing'
        if -1 not in set(trainData[col]):  # －1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
            max_interval = 5  # 分箱后的最多的箱数
            cutOff = ChiMerge(df=trainData, col=col, target='target', max_interval=max_interval, special_attribute=[], minBinPcnt=0)
            print cutOff


if __name__ == '__main__':
    main()

