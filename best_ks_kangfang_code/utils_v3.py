# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:55:20 2019
说明：常用评分卡模型使用函数，例如：卡方检验、单调性检验、KS计算 等等
@author: peng_zhang
"""

import pandas as pd
import numpy as np


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
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
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
        col_regroup[i + 1] = col_regroup[i] + col_regroup[i + 1]
        del_index.append(i)
        if direction == 'bad':
            if regroup['bad_rate'][i + 1] > 0:
                break
        else:
            if regroup['bad_rate'][i + 1] < 1:
                break
    col_regroup2 = [col_regroup[i] for i in range(len(col_regroup)) if i not in del_index]
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
    badRateEnconding = df[col].map(lambda x: br_dict[x])
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
    badRate = sum(df2[bad_col]) * 1.0 / sum(df2[total_col])
    df2['good'] = df2.apply(lambda x: x[total_col] - x[bad_col], axis=1)
    # df2['good'] = df2.apply(lambda x: x.total_col-x.bad_col, axis=1)
    goodRate = sum(df2['good']) * 1.0 / sum(df2[total_col])
    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df[total_col].apply(lambda x: x * badRate)
    df2['goodExpected'] = df[total_col].apply(lambda x: x * goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0] - i[1]) ** 2 / (i[0] + 0.00001) for i in badCombined]
    goodChi = [(i[0] - i[1]) ** 2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2


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
    if x <= cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin - 1)
    else:
        for i in range(0, numBin - 1):
            if cutOffPoints[i] < x <= cutOffPoints[i + 1]:
                return 'Bin {}'.format(i + 1)


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
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        if len(special_attribute) >= 1:
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))

        # 步骤一：通过col对数据集进行分组，求出每组的总样本数和坏样本数
        if N_distinct > 100:
            split_x = SplitData(df=df2, col=col, numOfSplit=100)
            # print(split_x)
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
        while len(groupIntervals) > split_intervals:  # 终止条件：当前分箱数==预设分箱数
            # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
            chisqList = []
            for k in range(len(groupIntervals) - 1):
                temp_group = groupIntervals[k] + groupIntervals[k + 1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad')
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined + 1]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[best_comnbined + 1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        #  检查是否没箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
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
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
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


def BadRateMonotone(df, sortByVar, target, special_attribute=[]):
    """
    功能：判断变量分箱后的坏样本率是否单调
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
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateNotMonotone = [badRate[i] < badRate[i+1] and badRate[i] < badRate[i-1] or
                          badRate[i] > badRate[i+1] and badRate[i] > badRate[i-1]
                          for i in range(1, len(badRate)-1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


# 计算IV值
def CalcWOE(df, col, target):
    """
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
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV': IV}
    # PYDEVD_USE_FRAME_EVAL = NO
    # 使用filter()：
    # https://oomake.com/question/1074868
    # >> > array
    # array([1., 2., 3., -Inf])
    # >> > sum(filter(lambda x: x != float('-inf'), array))
    # 6.0
    # IV = np.ma.masked_invalid(IV).sum()


def KS_AR(df, score, target):
    '''
    :param df: the dataset containing probability and bad indicator
    :param score:
    :param target:
    :return:
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total': total, 'bad': bad})
    all['good'] = all['total'] - all['bad']
    # all[score] = all.index
    all = all.sort_values(by=score, ascending=False)
    all.index = range(len(all))
    all.reset_index(level=0, inplace=True)
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    arList = [0.5 * all.loc[0, 'badCumRate'] * all.loc[0, 'totalPcnt']]
    for j in range(1, len(all)):
        ar0 = 0.5 * sum(all.loc[j - 1:j, 'badCumRate']) * all.loc[j, 'totalPcnt']
        arList.append(ar0)
    arIndex = (2 * sum(arList) - 1) / (all['good'].sum() * 1.0 / all['total'].sum())
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    # return {'AR': arIndex, 'KS': max(KS), 'cut_value': all.loc[KS.isin([max(KS)])][score].min()}
    return {'AR': arIndex, 'KS': max(KS)}
