# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:04:22 2019

@author: peng_zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


def BinBadRate(df, col, label, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param label: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby(col)[label].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby(col)[label].sum()
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

def BadRateMonotone(df, sortByVar, label,special_attribute = []):
    """
        判断某变量的坏样本率是否单调
        :param df: 包含检验坏样本率的变量，和目标变量
        :param sortByVar: 需要检验坏样本率的变量
        :param label: 目标变量，0、1表示好、坏
        :param special_attribute: 不参与检验的特殊值
        :return: 坏样本率单调与否
    """
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, label)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1]/x[0] for x in combined]
    badRateNotMonotone = [badRate[i] < badRate[i+1] and badRate[i] < badRate[i-1] or
                          badRate[i] > badRate[i+1] and badRate[i] > badRate[i-1]
                          for i in range(1, len(badRate) - 1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


def CalcWOE(df, col, label):
    """
        计算WOE和IV值
        :param df: 包含需要计算WOE的变量和目标变量
        :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
        :param label: 目标变量，0、1表示好、坏
        :return: 返回WOE和IV
    """
    total = df.groupby([col])[label].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[label].sum()
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


def MergeByCondition(x,condition_list):
    # condition_list是条件列表。满足第几个condition，就输出几
    s = 0
    for condition in condition_list:
        if eval(str(x) + condition):
            return s
        else:
            return s+1
    return s

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
    splitPointIndex = [int(i * n) for i in range(1, numOfSplit)]
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
        for i in range(N - 1):
            if bin[i] < x <= bin[i + 1]:
                return bin[i + 1]


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
    df2['good'] = df2.apply(lambda x: x[total_col]-x[bad_col], axis=1)
    goodRate = sum(df2['good'])*1.0 / sum(df2[total_col])
    df2['badExpected'] = df2[total_col].apply(lambda x: x*badRate)
    df2['goodExpected'] = df2[total_col].apply(lambda x: x*goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0]-i[1])**2 / (i[0] + 0.00001)for i in badCombined]
    goodChi = [(i[0]-i[1])**2 / (i[0] + 0.00001)for i in goodCombined]
    chisq = sum(badChi) + sum(goodChi)
    return chisq


def AssignBin(x, cutOffPoints, special_attribute=[]):
    """
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    """
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x) + 1
        return 'Bin {}'.format(0 - i)
    if x <= min(cutOffPoints):
        return 'Bin 0'
    elif x > max(cutOffPoints):
        return 'Bin {}'.format(numBin - 1)
    else:
        for i in range(0, numBin-1):
            if cutOffPoints[i] < x <= cutOffPoints[i+1]:
                return 'Bin {}'.format(i + 1)

def ChiMerge(df, col, label, max_interval=5, special_attribute=[], minBinPcnt=0.1):
    """
        卡方分箱
        :param df: 包含目标变量与分箱属性的数据框
        :param col: 需要分箱的属性
        :param label: 目标变量，取值0或1
        :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数
        :param special_attribute: 不参与分箱的属性取值
        :param minBinPcnt：最小箱的占比，默认为0
        :return: 分箱结果
    """
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct < max_interval:
        print("The number of original levels for {} is less than or equal to max intervals\n".format(col))
        return colLevels[:-1]
    else:
        if len(special_attribute) >= 1 :
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))
        # 步骤一：通过col对数据集进行分组，求出每组的总样本数和坏样本数
        if N_distinct > 100:
            split_x = SplitData(df2, col, numOfSplit=100)
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df2['temp'] = df2[col]
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', label, grantRateIndicator=1)

        # 首先，每个单独的属性值去重，将被分为单独的一组
        colLevels = sorted(list(set(df2['temp'])))
        # 对属性值进行排序，然后两两组别进行合并
        groupIntervals = [[i] for i in colLevels]

        #  步骤二建立循环，不断合并最优的相邻两个组别，直到
        #  1，最终分裂出来的分箱数<＝预设的最大分箱数
        #  2，每箱的占比不低于预设值（可选）
        #  3，每箱同时包含好坏样本
        #  如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        split_intervals = max_interval - len(special_attribute)
        while len(groupIntervals) > split_intervals: # 终止条件：当前分箱数==预设分箱数
            chisqList = []
            for k in range(len(groupIntervals) - 1):
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad')
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined +1]
            groupIntervals.remove(groupIntervals[best_comnbined+1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        #  检查每箱是否都有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
        groupdvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
        df2['temp_Bin'] = groupdvalues
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', 'label')
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
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', 'label')
                chisq1 = Chi2(df2b, 'total', 'bad')
                # 和后一箱进行合并，并且计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', label)
                chisq2 = Chi2(df2b, 'total', 'bad')
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])

            # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupdvalues = df2['temp'].apply(lambda x: AssignBin(x=x, cutOffPoints=cutOffPoints))
            df2['temp_Bin'] = groupdvalues
            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', label)
            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]

        # 需要检查分箱后的最小占比
        if minBinPcnt > 0:
            groupdvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupdvalues
            valuesCounts =  groupdvalues.value_counts().to_frame()
            N = sum(valuesCounts['temp'])
            valuesCounts['pcnt'] = valuesCounts['temp'].apply(lambda x: x * 1.0 / N)
            valuesCounts = valuesCounts.sort_index()
            minPcnt = min(valuesCounts['pcnt'])
            while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
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
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', label)
                    chisq1 = Chi2(df2b, 'total', 'bad')

                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valuesCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', label)
                    # chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                    chisq2 = Chi2(df2b, 'total', 'bad')
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])

                groupdvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
                df2['temp_Bin'] = groupdvalues
                valuesCounts = groupdvalues.value_counts().to_frame()
                N = sum(valuesCounts['temp'])
                valuesCounts['pcnt'] = valuesCounts['temp'].apply(lambda x: x * 1.0 / N)
                valuesCounts = valuesCounts.sort_index()
                minPcnt = min(valuesCounts['pcnt'])
        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints


def BadRateMonotone(df, sortByVar, label, special_attribute=[]):
    """
    功能：判断变量分箱后的坏样本率是否单调
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param label: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    """
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True

    regroup = BinBadRate(df2, sortByVar, label)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]

    badRateNotMonotone = [badRate[i] < badRate[i+1] and badRate[i] < badRate[i-1] or
                          badRate[i] > badRate[i+1] and badRate[i] > badRate[i-1]
                          for i in range(1, len(badRate)-1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


def KS(df, score, target):
    """
        计算KS值
        :param df: 包含目标变量与预测值的数据集
        :param score: 得分或者概率
        :param target: 目标变量
        :return: KS值
    """
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    # all[score] = all.index
    all.reset_index(drop=True)
    all = all.sort_values(by=score, ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(KS)


def Prob2Score(prob, basePoint, PDO):
    """
        将概率映射成分数
        :param prob:
        :param basePoint:
        :param PDO:
        :return:
    """
    y = np.log(prob/(1-prob))
    score = int(basePoint + PDO/np.log(2)*-y)
    return score


#################################
#   1, 读取数据，衍生初始变量   #
#################################
def main():
    folderOfData = 'D:/financialStudy/Bcard/data/'
    trainData = pd.read_csv(folderOfData + 'trainData.csv', header=0)
    testData = pd.read_csv(folderOfData + 'testData.csv', header=0)
    allFeatures = []
    '''
    逾期类型的特征在行为评分卡（预测违约行为）中，一般是非常显著的变量。
    通过设定时间窗口，可以衍生以下类型的逾期变量：
    '''
    # 考虑过去1个月，3个月，6个月，12个月
    for t in [1, 3, 6, 12]:
        # 1，过去t时间窗口内的最大逾期状态
        allFeatures.append('maxDelqL' + str(t) + 'M')
        trainData['maxDelqL' + str(t) + 'M'] = trainData.apply(lambda x: DelqFeatures(x, t, 'max delq'), axis=1)

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
        allFeatures.append('maxUrateL' + str(t) + 'M')
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
        trainData['maxPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'max payment ratio'),
                                                              axis=1)

        # 2，过去t时间窗口内的最小月还款率
        allFeatures.append('minPayL' + str(t) + "M")
        trainData['minPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'min payment ratio'),
                                                              axis=1)

        # 3，过去t时间窗口内的平均月还款率
        allFeatures.append('avgPayL' + str(t) + "M")
        trainData['avgPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'mean payment ratio'),
                                                              axis=1)

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
    small_bin_var = []
    large_bin_var = []
    N = trainData.shape[0]

    for var in categoricalFeatures:
        if var not in not_monotone:
            total = trainData.groupby(var)[var].count()
            pcnt = total * 1.0 / N
            if min(pcnt) <= 0.05:
                small_bin_var.append({var: pcnt.to_dict()})
            else:
                large_bin_var.append(var)

    # 直接剔除类别型特征某个分箱占比于0.9以上的变量
    for i in range(len(small_bin_var)):
        for k in small_bin_var[i].values():
            for g in k.values():
                if g > 0.9:
                    small_bin_var.pop(i)

    trainData['maxDelqL1M_Bin'] = trainData['maxDelqL1M'].apply(lambda x: MergeByCondition(x, ['==0', '==1', '>=2']))
    trainData['maxDelqL3M_Bin'] = trainData['maxDelqL3M'].apply(lambda x: MergeByCondition(x, ['==0', '==1', '>=2']))
    trainData['maxDelqL6M_Bin'] = trainData['maxDelqL6M'].apply(lambda x: MergeByCondition(x, ['==0', '==1', '>=2']))
    for var in ['maxDelqL1M_Bin', 'maxDelqL3M_Bin', 'maxDelqL6M_Bin']:
        WOE_IV_dict[var] = CalcWOE(trainData, var, 'label')

    '''
    对于数值型变量，需要先分箱，再计算WOE、IV
    分箱的结果需要满足：
    1，箱数不超过5
    2，bad rate单调
    3，每箱占比不低于5%
    '''
    bin_dict = []
    for var in numericalFeatures:
        binNum = 5
        newBin = var + '_Bin'
        bin = ChiMerge(trainData, var, 'label', max_interval=binNum, minBinPcnt=0.05)
        trainData[newBin] = trainData[var].apply(lambda x: AssignBin(x, bin))
        while not BadRateMonotone(trainData, newBin, 'label'):
            binNum -= 1
            bin = ChiMerge(trainData, var, 'label', max_interval=binNum, minBinPcnt=0.05)
            trainData[newBin] = trainData[var].apply(lambda x: AssignBin(x, bin))
        WOE_IV_dict[newBin] = CalcWOE(trainData, newBin, 'label')
        bin_dict.append({var: bin})

    ##############################
    #   3, 单变量分析和多变量分析   #
    ##############################
    #  选取IV高于0.02的变量

    high_IV = [(k, v['IV']) for k, v in WOE_IV_dict.items() if v['IV'] > 0.02]

    high_IV_sorted = sorted(high_IV, key=lambda k: k[1], reverse=True)

    for (var, iv) in high_IV_sorted:
        newVar = var + '_WOE'
        trainData[newVar] = trainData[var].map(lambda x: WOE_IV_dict[var]['WOE'][x])

    '''
    多变量分析：比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
    '''
    deleted_index = []
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0]+'_WOE'
        for j in range(cnt_vars):
            if i == j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0]+'_WOE'
            roh = np.corrcoef(trainData[x1], trainData[y1])[0, 1]
            if abs(roh) > 0.7:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)

    single_analysis_vars = [high_IV_sorted[i][0] + "_WOE" for i in range(cnt_vars) if i not in deleted_index]

    X = trainData[single_analysis_vars]
    f, ax = plt.subplots(figsize=(10, 10))
    corr = X.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap="YlGnBu", square=True, linewidths=0.5, vmax=1, vmin=0, annot=True,
                annot_kws={'size': 6, 'weight': 'bold'}, ax=ax)
    plt.subplots_adjust(left=0.25, wspace=0.25, hspace=0.25, bottom=0.3, top=0.9)

    '''
    多变量分析：VIF
    '''
    X = np.matrix(trainData[single_analysis_vars])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    print(max(VIF_list))

    # 最大的VIF是 3.429，小于10，因此这一步认为没有多重共线性
    multi_analysis = single_analysis_vars

    ################################
    #   4, 建立逻辑回归模型预测违约   #
    ################################
    X = trainData[multi_analysis]
    X['intercept'] = [1] * X.shape[0]
    y = trainData['label']

    import statsmodels.api as sm
    logit = sm.Logit(y, X)
    logit_result = logit.fit()
    pvalues = logit_result.pvalues
    params = logit_result.params
    fit_result = pd.concat([params, pvalues], axis=1)
    fit_result.columns = ['coef', 'p-value']
    fit_result = fit_result.sort_values(by='coef')
    '''
                                    coef        p-value
        intercept                 -1.814153   0.000000e+00
        increaseUrateL6M_Bin_WOE  -1.179169   1.091337e-60
        M1FreqL3M_Bin_WOE         -0.789777  5.503151e-214
        avgUrateL1M_Bin_WOE       -0.690117   5.899962e-16
        M2FreqL3M_Bin_WOE         -0.662967   3.786592e-61
        avgPayL3M_Bin_WOE         -0.483425   3.323171e-39
        avgUrateL3M_Bin_WOE       -0.459164   1.879837e-05
        avgUrateL6M_Bin_WOE       -0.191476   2.470378e-01
        maxUrateL6M_Bin_WOE       -0.135591   3.010442e-01
        M1FreqL6M_Bin_WOE         -0.133468   1.453110e-04
        maxPayL12M_Bin_WOE        -0.109359   3.239076e-01
        increaseUrateL12M_Bin_WOE -0.036373   7.207816e-01
        avgPayL6M_Bin_WOE         -0.032356   4.559436e-01
        maxPayL6M_Bin_WOE          0.014441   8.353810e-01
        M0FreqL12M_Bin_WOE         0.048129   3.261829e-01
        avgUrateL12M_Bin_WOE       0.054752   7.724990e-01
        avgPayL1M_Bin_WOE          0.059621   3.757585e-01
        minPayL6M_Bin_WOE          0.113984   2.863916e-01
        minPayL3M_Bin_WOE          0.272055   5.843495e-06
    '''


    for var in list(fit_result[fit_result['coef']>0].index):
        print(sm.Logit(y, trainData[var]).fit().params)

        # maxPayL6M_Bin_WOE - 1.081271
        # M0FreqL12M_Bin_WOE - 1.016628
        # avgUrateL12M_Bin_WOE - 1.026815
        # avgPayL1M_Bin_WOE - 0.985094
        # minPayL6M_Bin_WOE - 0.808652
        # minPayL3M_Bin_WOE - 0.828549

    # 单独建立回归模型，系数为负，与预期相符，说明仍然存在多重共线性
    # 下一步，用GBDT跑出变量重要性，挑选出合适的变量
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    gbdt_model = clf.fit(X, y)
    importace = gbdt_model.feature_importances_.tolist()
    featureImportance = zip(multi_analysis, importace)
    featureImportanceSorted = sorted(featureImportance, key=lambda k: k[1], reverse=True)

    # 先假定模型可以容纳5个特征，再逐步增加特征个数，直到有特征的系数为正，或者p值超过0.1
    n = 5
    featureSelected = [i[0] for i in featureImportanceSorted[:n]]
    X_train = X[featureSelected + ['intercept']]
    logit = sm.Logit(y, X_train)
    logit_result = logit.fit()
    pvalues = logit_result.pvalues
    params = logit_result.params
    fit_result = pd.concat([params, pvalues], axis=1)
    fit_result.columns = ['coef', 'p-value']
    '''
                                    coef       p-value
        M1FreqL3M_Bin_WOE        -0.838782  0.000000e+00
        increaseUrateL6M_Bin_WOE -1.190850  9.668511e-94
        avgUrateL1M_Bin_WOE      -1.032359  1.829976e-55
        avgPayL3M_Bin_WOE        -0.429920  2.297665e-67
        M2FreqL3M_Bin_WOE        -0.689150  1.611216e-67
        intercept                -1.815944  0.000000e+00
    '''
    while (n<len(featureImportanceSorted)):
        nextVar = featureImportanceSorted[n][0]
        featureSelected = featureSelected + [nextVar]
        X_train = X[featureSelected+['intercept']]
        logit = sm.Logit(y, X_train)
        logit_result = logit.fit()
        params = logit_result.params
        if max(params) < 0:
            n += 1
        else:
            featureSelected.remove(nextVar)
            n += 1
    """
        M1FreqL3M_Bin_WOE           -0.773008
        increaseUrateL6M_Bin_WOE    -1.160523
        avgUrateL1M_Bin_WOE         -0.691099
        avgPayL3M_Bin_WOE           -0.415675
        M2FreqL3M_Bin_WOE           -0.665428
        M1FreqL6M_Bin_WOE           -0.127457
        avgUrateL3M_Bin_WOE         -0.499585
        maxUrateL6M_Bin_WOE         -0.179151
        avgUrateL12M_Bin_WOE        -0.038569
        maxPayL12M_Bin_WOE          -0.028079
        increaseUrateL12M_Bin_WOE   -0.044080
        avgPayL6M_Bin_WOE           -0.010414
        intercept                   -1.810605
    """
    X_train = X[featureSelected+['intercept']]
    logit = sm.Logit(y, X_train)
    logit_result = logit.fit()
    pvalues = logit_result.pvalues
    params = logit_result.params
    fit_result = pd.concat([params, pvalues], axis=1)
    fit_result.columns = ['coef', 'p-value']
    fit_result.sort_values(by='p-value', ascending=False)

    # 存放p-value大于0.1的变量

    largePValueVars = pvalues[pvalues > 0.1].index
    for var in largePValueVars:
        X_temp = X[[var, 'intercept']]
        logit = sm.Logit(y, X_temp)
        logit_result = logit.fit()
        pvalues = logit_result.pvalues
        print('The p-value of {} is {}'.format(var, pvalues[var]))

    """
        The p-value of maxUrateL6M_Bin_WOE is 1.0220855208089309e-35
        The p-value of avgUrateL12M_Bin_WOE is 1.0074949683072229e-18
        The p-value of maxPayL12M_Bin_WOE is 1.700886664115416e-46
        The p-value of increaseUrateL12M_Bin_WOE is 1.9777915300948074e-57
        The p-value of avgPayL6M_Bin_WOE is 0.0
    """

    '''
        显然，单个变量的p值是显著地。说明任然存在着共线性。
        可用L1约束，直到所有变量显著
    '''
    X2 = X[featureSelected+['intercept']]
    for alpha in range(100, 0, -1):
        l1_logit = sm.Logit.fit_regularized(sm.Logit(y, X2), start_params=None, method='l1', alpha=alpha)
        pvalues = l1_logit.pvalues
        params = l1_logit.params
        if max(params) > 0 or max(pvalues) >= 0.1:
            break
    bestAlpha = alpha + 1
    l1_logit = sm.Logit.fit_regularized(sm.Logit(y, X2), start_params=None, method='l1', alpha=bestAlpha)
    params2 = l1_logit.params.to_dict()

    featuresInModel = [k for k, v in params2.items() if k != 'intercept' and v < -0.0000001]

    # 最后的入模训练
    X_train = X[featuresInModel + ['intercept']]
    logit = sm.Logit(y, X_train)
    logit_result = logit.fit()
    pvalues = logit_result.pvalues
    params = logit_result.params
    fit_result = pd.concat([params, pvalues], axis=1)
    fit_result.columns = ['coef', 'p-value']
    fit_result.sort_values(by='p-value', ascending=False)

    '''
                                      coef        p-value
            M1FreqL3M_Bin_WOE        -0.771000  5.846552e-228
            increaseUrateL6M_Bin_WOE -1.166847   4.940995e-89
            avgUrateL1M_Bin_WOE      -0.728788   3.043448e-19
            avgPayL3M_Bin_WOE        -0.421456   4.391168e-64
            M2FreqL3M_Bin_WOE        -0.665851   5.134180e-62
            M1FreqL6M_Bin_WOE        -0.132967   1.375606e-05
            avgUrateL3M_Bin_WOE      -0.550823   2.445287e-10
            intercept                -1.810598   0.000000e+00
    '''
    from sklearn.metrics import roc_auc_score
    trainData['train_pred'] = logit_result.predict(X_train)
    train_ks = KS(trainData, 'train_pred', 'label')
    train_auc = roc_auc_score(trainData['label'], trainData['train_pred'])

    ###################################
    #   5，在测试集上测试逻辑回归的结果   #
    ###################################
    # 准备WOE编码后的变量

    modelFeatures = [var.replace('_Bin_WOE', '') for var in featuresInModel]

    numFeatures = [i for i in modelFeatures if i in numericalFeatures]
    charFeatures = [i for i in modelFeatures if i in categoricalFeatures]

    testData['maxDelqL1M'] = testData.apply(lambda x: DelqFeatures(x, 1, 'max delq'), axis=1)
    testData['maxDelqL3M'] = testData.apply(lambda x: DelqFeatures(x, 3, 'max delq'), axis=1)
    testData['M2FreqL3M'] = testData.apply(lambda x: DelqFeatures(x, 3, 'M2 times'), axis=1)
    testData['M0FreqL3M'] = testData.apply(lambda x: DelqFeatures(x, 3, 'M0 times'), axis=1)
    testData['M1FreqL6M'] = testData.apply(lambda x: DelqFeatures(x, 6, 'M1 times'), axis=1)
    testData['M2FreqL3M'] = testData.apply(lambda x: DelqFeatures(x, 3, 'M2 times'), axis=1)
    testData['M1FreqL12M'] = testData.apply(lambda x: DelqFeatures(x, 12, 'M1 times'), axis=1)
    testData['maxUrateL6M'] = testData.apply(lambda x: UrateFeatures(x,6,'max utilization rate'),axis = 1)
    testData['avgUrateL1M'] = testData.apply(lambda x: UrateFeatures(x, 1, 'mean utilization rate'), axis=1)
    testData['avgUrateL3M'] = testData.apply(lambda x: UrateFeatures(x, 3, 'mean utilization rate'), axis=1)
    testData['avgUrateL6M'] = testData.apply(lambda x: UrateFeatures(x,6, 'mean utilization rate'),axis=1)
    testData['increaseUrateL6M'] = testData.apply(lambda x: UrateFeatures(x, 6, 'increase utilization rate'), axis=1)
    testData['avgPayL3M'] = testData.apply(lambda x: PaymentFeatures(x, 3, 'mean payment ratio'),axis=1)
    testData['avgPayL6M'] = testData.apply(lambda x: PaymentFeatures(x, 6, 'mean payment ratio'),axis=1)
    testData['M1FreqL3M'] = testData.apply(lambda x: DelqFeatures(x, 3, 'M1 times'), axis=1)

    testData['M2FreqL3M_Bin'] = testData['M2FreqL3M'].apply(lambda x: int(x >= 1))
    testData['M1FreqL3M'] = testData['M1FreqL3M'].apply(lambda x: int(x >= 1))
    testData['maxDelqL1M_Bin'] = testData['maxDelqL1M'].apply(lambda x: MergeByCondition(x, ['==0', '==1', '>=2']))
    testData['maxDelqL3M_Bin'] = testData['maxDelqL3M'].apply(lambda x: MergeByCondition(x, ['==0', '==1', '>=2']))

    for var in numFeatures:
        newBin = var +'_Bin'
        bin = [list(i.values())[0] for i in bin_dict if var in i.keys()][0]
        testData[newBin] = testData[var].apply(lambda x: AssignBin(x, bin))

    finalFeature = [i + '_Bin' for i in numFeatures+charFeatures]
    for var in finalFeature:
        var2 = var + '_WOE'
        testData[var2] = testData[var].map(lambda x: WOE_IV_dict[var]['WOE'][x])

    X_test = testData[featuresInModel]
    X_test['intercept'] = [1]*testData.shape[0]
    y_test = testData['label']
    logit = sm.Logit(y_test, X_test)
    logit_result = logit.fit()
    testData['test_pred'] = logit_result.predict(X_test)
    test_ks = KS(testData, 'test_pred', 'label')
    test_auc = roc_auc_score(testData['label'], testData['test_pred'])

    ##########################
    #   6，在测试集上计算分数   #
    ##########################
    basePoint = 500
    PDO = 20
    testData['score'] = testData['test_pred'].apply(lambda x:Prob2Score(x, basePoint=basePoint, PDO=PDO))
    plt.hist(testData['score'], bins=100)

if __name__ == '__main__':
    main()
