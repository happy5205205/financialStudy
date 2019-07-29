# _*_ coding: utf-8 _*_

import pandas as pd
import numpy as np
import os
import re
import datetime
import time
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示符号

def CareerYear(x):
    # 对工作年限进行修改
    x = str(x)
    if x.find("-1") == 0:
        return -1
    elif x.find('10+') == 0:
        return 11
    elif x.find('< 1') == 0:
        return 0.5
    else:
        c = re.sub(r'\D', "", x)
        return c


def DescExisting(x):

    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConverDataStr(x):
    """
        对日期进行处理
        :param x:
        :return:
    """
    mth_dict={'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    # time.mktime 不能读取1970年之前的日期
    if x == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1', '%Y-%m')))
    else:
        yr = int(x[4:6])
        if yr < 17:
            yr = 2000 + yr
        else:
            yr = 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr, mth, 1)


def MakeupMissing(x):
    """
        对缺失值处理
        :param x:
        :return:
    """
    if np.isnan(x):
        return -1
    else:
        return x


def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate, earlyDate)
        return gap.years * 12 + gap.months
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
    total = df.groupby(col)[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby(col)[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = pd.merge(total, bad, how='left', right_index=True, left_index=True)
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator == 0:
        return (dicts, regroup)
    else:
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        overallRate = B*1.0 / N

        return (dicts, regroup, overallRate)


def MergeBad0(df, col, target, direction='bad'):
    """
     类别型变量分组后存在好或则坏比为0时，需要合并分组
     :param df: 包含检验0％或者100%坏样本率
     :param col: 分箱后的变量或者类别型变量。检验其中是否有一组或者多组没有坏样本或者没有好样本。如果是，则需要进行合并
     :param target: 目标变量，0、1表示好、坏
     :return: 合并方案，使得每个组里同时包含好坏样本
    """
    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]

    if direction=='bad':
        # 如果是合并坏样本率为0的组，则跟最小的坏样本率为非0的组进行合并
        regroup = regroup.sort_values(by='bad_rate')
    else:
        # 如果是合并好样本率为0的组，则跟最小的好样本率为非0的组进行合并
        regroup = regroup.sort_values(by='bad_rate',ascending=False)
    regroup.index = range(regroup.shape[0])
    col_regroup = [[i] for i in regroup[col]]
    del_index = []
    col_regroup = [[i] for i in regroup[col]]
    for i in range(regroup.shape[0] -1):
        col_regroup[i + 1] = col_regroup[i] + col_regroup[i + 1]
        del_index.append(i)
        if direction=='bad':
            if regroup['bad_rate'][i+1] > 0:
                break
        else:
            if regroup['bad_rate'][i+1] < 1:
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
    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]
    br_dict = regroup[[col, 'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding': badRateEnconding, 'bad_rate': br_dict}

"""
    分界线，下列函数全用于卡方分箱函数
"""


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

def ChiMerge(df, col, target, max_interval=5, special_attribute=[], minBinPcnt=0.1):
    """
        卡方分箱
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
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)

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
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', 'target')
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
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', 'target')
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

                groupdvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
                df2['temp_Bin'] = groupdvalues
                valuesCounts = groupdvalues.value_counts().to_frame()
                N = sum(valuesCounts['temp'])
                valuesCounts['pcnt'] = valuesCounts['temp'].apply(lambda x: x * 1.0 / N)
                valuesCounts = valuesCounts.sort_index()
                minPcnt = min(valuesCounts['pcnt'])
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
    df3 = pd.DataFrame()

    badRateNotMonotone = [badRate[i] < badRate[i+1] and badRate[i] < badRate[i-1] or
                          badRate[i] > badRate[i+1] and badRate[i] > badRate[i-1]
                          for i in range(1, len(badRate)-1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


def CalcWOE(df, col, target):
    """
        计算IV值
        :param df: 包含需要计算WOE的变量和目标变量
        :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
        :param target: 目标变量，0、1表示好、坏
        :return: 返回WOE和IV
    """
    total = df.groupby(col)[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby(col)[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = pd.merge(total, bad, how='left', right_index=True, left_index=True)
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    G = N - B
    regroup['good'] = regroup.apply(lambda x: x.total-x.bad, axis=1)
    # regroup['bad_pcnt'] = regroup.apply(lambda x: x.bad*1.0/B)
    # regroup['good_pcnt'] = regroup.apply(lambda x: x.good*1.0/G)
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt/x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt), axis=1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV': IV}

def Prob2Score(prob, basePoint, PDO):
    """
        将概率映射成分数
        :param prob:
        :param basePoint:
        :param PDO:
        :return:
    """
    y =  np.log(prob/(1-prob))
    score = int(basePoint + PDO/np.log(2)*(-y))
    return score


def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    regroup = pd.DataFrame({'total':total, 'bad':bad})
    regroup['good'] = regroup['total'] - regroup['bad']
    regroup[score] = regroup.index
    regroup = regroup.sort_values(by=score, ascending=False)
    regroup.index = range(len(regroup))
    regroup['badCumRate'] = regroup['bad'].cumsum()/regroup['bad'].sum()
    regroup['goodCumRate'] = regroup['good'].cumsum()/regroup['good'].sum()
    ks_value = regroup.apply(lambda x:x.badCumRate-x.goodCumRate, axis=1)
    return max(ks_value)


def Gini(df, score, target):
    """

        :param df: 数据集
        :param score: 概率或分数二选一
        :param target: 目标变量
        :return:
    """
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    regroup = pd.DataFrame({'total': total, 'bad': bad})
    regroup[score] = regroup.index
    # regroup.reset_index(drop=True)
    regroup = regroup.sort_values(by=score, ascending=False)
    regroup.index = range(len(regroup))
    regroup['badCumRate'] = regroup['bad'].cumsum() / regroup['bad'].sum()
    regroup['totalCumRate'] = regroup['total']/ regroup['total'].sum()
    regroup['rate'] = (1-regroup['badCumRate'])*regroup['badCumRate']
    regroup['ginirate'] = regroup['rate']*regroup['totalCumRate']
    # gini_value = regroup.apply(lambda x: (x.total/sum(x.total)) - (1-x.badCumRate + 0.00001)*x.badCumRate, axis=1)
    # gini_value = regroup.apply(lambda x: (regroup['total']/sum(regroup['total'])) * ((1 - regroup['badCumRate']) * regroup['badCumRate']))
    return sum(regroup['ginirate'])


def psi(actual, expected, title='PSI', quant=10):
	"""
        衡量测试样本与模型开发样本评分的分布差异
        param actual: 实际输出概率
        param expected: 预期输出概率
        param title: 名称
        param quant: 将输出概率按照10等分

        公式: PSI = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
        步骤：
        1、训练按输出概率p1从小到大排序10等分
        2、在新样本的预测中，输出概率p2从小到大排序，按p1区间10等分
        3、计算。其中实际占比：p2上各区间的用户占比
        预期占比：p1上各区间的用户占比
        psi<0.1 稳定性很高
        0.1<=psi<=0.25 稳定性一般
        psi>0.25 稳定性很差，重做
	"""
	plt.rcParams['font.sans-serif']=['SimHei']  # 显示中文
	plt.rcParams['axes.unicode_minus']=False  # 正常显示符号
	min_v = min(min(actual['train_pred']), min(expected['test_pred']))  # 两样本输出概率的最小值
	max_v = max(max(actual['train_pred']), max(expected['test_pred']))  # 两样本输出概率的最大值
	interval = 1.0*(max_v-min_v)/quant
	acnt = []
	ecnt = []
	s, e = min_v, min_v + interval
	act = np.array(actual)
	expe = np.array(expected)
	# 分组
	while float(e) <= max_v:
		acnt.append(((act >= s) & (act < e)).sum())
		ecnt.append(((expe >= s) & (expe < e)).sum())
		s = e
		e = e + interval
	arate = np.array(acnt) / len(actual)
	erate = np.array(ecnt) / len(expected)
	arate[arate==0] = 0.000001
	erate[erate==0] = 0.000001
	# 计算psi值
	psi = np.sum((arate - erate)*np.log(arate/erate))
	# 画图
	x1 = np.linspace(0, len(acnt)-1, len(acnt)) -0.2
	x2 = np.linspace(0, len(acnt)-1, len(acnt)) +0.2
	plt.bar(x1, arate*100, alpha=0.9, width=0.4, facecolor="orange", edgecolor="white")
	plt.bar(x2, erate*100, alpha=0.9, width=0.4, facecolor="lightblue", edgecolor="white")
	#plt.legend() # 显示网格
	plt.suptitle(title)
	plt.show()
	return psi


def lift_chart(df):
    """
        提升图
    """
    # plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
    # plt.rcParams["axes.unicode_minus"] = False  # 正常显示符号
    df = df.sort_values(by="test_pred", ascending=False)  # 按输出概率降序
    df = df.reset_index(drop=True)
    # 计算占比
    rows = []
    for group in np.array_split(df, 10):
        Sum_defaults = group['target'].sum()
        rows.append({'total': len(group), 'bad': Sum_defaults})
    lift = pd.DataFrame(rows)
    lift['obs'] = 100
    lift['BadCapturedByModel'] = lift['bad'] / lift['bad'].sum()
    lift['BadCapturedRandomly'] = 10 / lift['obs']
    lift['CumulativeBadByModel'] = np.nan
    lift['CumulativeBadRandomly'] = np.nan
    model_bad_list = list(lift['BadCapturedByModel'])
    for i in range(len(model_bad_list)):
        lift['CumulativeBadByModel'][i] = sum(model_bad_list[:i + 1])

    random_bad_list = list(lift['BadCapturedRandomly'])
    for i in range(len(random_bad_list)):
        lift['CumulativeBadRandomly'][i] = sum(random_bad_list[:i + 1])
    lift['Lift'] = lift['CumulativeBadByModel'] / lift['CumulativeBadRandomly']
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    bar_width = 0.3
    plt.bar(x=range(1,len(model_bad_list)+1), height=list(lift['BadCapturedByModel']),
            label='BadCapturedByModel', alpha=0.8, width=bar_width)
    plt.bar(x=np.arange(1, len(random_bad_list)+1) + bar_width, height=list(lift['BadCapturedRandomly']),
            label='BadCapturedRandomly', alpha=0.8, width=bar_width)
    # for x, y in enumerate(list(lift['BadCapturedByModel'])):
    #     plt.text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=9)
    # for x, y in enumerate(list(lift['BadCapturedRandomly'])):
    #     plt.text(x + bar_width, y, '%.2f' % y, ha='center', va='top', fontsize=9)
    plt.title('Lift Chart')
    plt.xlabel('Decile')
    plt.ylabel('BadRate')

    plt.subplot(1,2,2)
    plt.plot(lift['Lift'], marker='o')
    plt.title('Cumlative Lift Chart')
    plt.xlabel('Decile')
    plt.ylabel('Lift')
    plt.grid(True)
    plt.savefig('Lift.jpg')
    plt.close()
    return lift


def draw_lorenz(df):
	"""洛伦兹曲线
	"""
	plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
	plt.rcParams["axes.unicode_minus"] = False  # 正常显示符号
	df = df.sort_values(by="test_pred", ascending=False)  # 按输出概率降序
	df = df.reset_index(drop=True)
	Y = df.target
	Y_lorenz = 100*(Y.cumsum()/Y.sum())  # 累计占比
	Y_lorenz = np.array(Y_lorenz)
	x = list(range(0, len(Y)))
	x = np.array(x)
	x = 100*x/len(Y)
	plt.axis([0, 100, 0, 100])
	st_line = np.arange(0, 101, 1)
	plt.plot(x, Y, st_line)
	plt.xlabel(u"累计单量占比")
	plt.ylabel(u"累计违约占比")
	plt.title(u"洛伦兹曲线")
	plt.grid(True)  # 显示网格
	plt.show()


def main():
    data_path = './data'
    out_path = './result'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for decode in ('gbk', 'utf-8', 'gb18030'):
        try:
            allData = pd.read_csv(os.path.join(data_path, 'application.csv'), encoding=decode, error_bad_lines=False)
            print('data-' + decode + '-success!!')
            break
        except:
            pass
    # allData = pd.read_csv(os.path.join(data_path, 'application.csv'), encoding='latin1')
    allData['term'] = allData['term'].apply(lambda x: int(x.replace('months', '')))
    # target 处理，loan_status标签中Fully Paid是正常客户  Charged Off是违约客户
    allData['target'] = allData['loan_status'].map(lambda x: int(x == 'Charged Off'))
    '''
        由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
    '''
    allData = allData[allData['term'] == 36]

    '''
        第一步数据预处理
        1、数据清洗
        2、格式转换
        3、缺失值处理

    '''
    print('数据预处理开始')
    # 将带%的百分比变为浮点数
    allData['int_rate_clean'] = allData['int_rate'].apply(lambda x: float(x.replace('%', ''))/100)

    # 将工作年限进行转换
    # allData['emp_length_clean'] = allData['emp_length'].apply(CareerYear)
    allData['emp_length'].fillna('-1', inplace=True)
    allData['emp_length_clean'] = allData['emp_length'].map(CareerYear)
    allData['emp_length_clean'] = allData['emp_length_clean'].apply(lambda x: int(x))

    # 将desc的缺失作为一种状态，非缺失作为另一种状态
    allData['desc_clean'] = allData['desc'].map(DescExisting)

    # 处理日期 earliest_cr_line标签的格式不统一，需要统一格式且转换成python格式
    allData['earliest_cr_line_clean'] = allData['earliest_cr_line'].map(ConverDataStr)

    allData['app_date_clean'] = allData['issue_d'].map(ConverDataStr)

    # 对缺失值进行处理
    allData['mths_since_last_delinq_clean'] = allData['mths_since_last_delinq'].map(MakeupMissing)
    allData['mths_since_last_record_clean'] = allData['mths_since_last_record'].map(MakeupMissing)
    allData['pub_rec_bankruptcies_clean'] = allData['pub_rec_bankruptcies'].map(MakeupMissing)
    print('数据预处理完成')

    """
        第二步：变量的衍生
    """
    print('变量的衍生开始')
    # 考虑申请额度于收入的占比
    allData['limit_income'] = allData.apply(lambda x: x.loan_amnt / x.annual_inc, axis=1)
    # 考虑earliest_cr_line到申请日期的跨度，以月份记
    allData['earliest_cr_to_app'] = allData.apply(lambda x: MonthGap(x.earliest_cr_line_clean, x.app_date_clean), axis=1)
    print('变量的衍生完成')

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

    print('类别型变量分箱')
    print('类别型变量取值小于5的变量分箱开始')
    more_value_feature = [col for col in cat_features if len(set(allData[col])) > 5]  # 存放变量取值大于5
    less_value_feature = [col for col in cat_features if len(set(allData[col])) <= 5]  # 存放变量取值少于5

    # (i)取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
    merge_bin_dict = {}  # 存放需要合并的变量，以及合并方法
    var_bin_list = []  # 由于某个取值没有好或者坏样本而需要合并的变量
    for col in less_value_feature:
        binBadRate = BinBadRate(allData, col, 'target')[0]

        if min(binBadRate.values()) == 0:
            print('{}标签中存在坏样本比例为0，需要合并'.format(col))
            combine_bin = MergeBad0(allData, col, 'target')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            allData[newVar] = allData[col].map(combine_bin)
            var_bin_list.append(newVar)
        if max(binBadRate.values()) == 1:
            print('{}标签中存在好样本比例为0，需要合并'.format(col))
            combine_bin = MergeBad0(allData, col, 'target')
            merge_bin_dict[col] = combine_bin
            newVar = col + 'Bin'
            allData[newVar] = allData[col].map(combine_bin)
            var_bin_list.append(newVar)

    # less_value_feature中剔除需要合并的变量
    less_value_feature = [i for i in less_value_feature if i+'_Bin' not in var_bin_list]
    print('类别型变量取值小于5的变量分箱结束')
    print('类别型变量分箱')

    # （ii）当取值>5时：用bad rate进行编码，放入连续型变量里
    br_encoding_dict = {}  # 记录按照bad rate进行编码的变量，及编码方式
    print('类别型变量取值大于5的变量编码开始')
    for col in more_value_feature:
        br_encoding = BadRateEncoding(allData, col, 'target')
        allData[col+'_br_encoding'] = br_encoding['encoding']
        br_encoding_dict[col] = br_encoding['bad_rate']
        num_features.append(col+'_br_encoding')
    print('类别型变量取值大于5的变量编码结束')

    # （iii）对连续型变量进行分箱，包括（ii）中的变量
    continous_merged_dict = {}
    for col in num_features:
        # －1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
        if -1 not in set(allData[col]):
            # 分箱后的最多的箱数
            max_interval = 5
            cutOff = ChiMerge(allData, col, 'target', max_interval=max_interval, special_attribute=[], minBinPcnt=0)
            allData[col+'_Bin'] = allData[col].apply(lambda x: AssignBin(x, cutOff, special_attribute=[]))
            monotone = BadRateMonotone(allData, col+'_Bin', 'target')
            # print(monotone)
            while (not monotone):
                # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                max_interval -= 1
                cutOff = ChiMerge(allData, col, 'target', max_interval=max_interval, special_attribute=[],minBinPcnt=0)
                allData[col + '_Bin'] = allData[col].apply(lambda x: AssignBin(x, cutOff, special_attribute=[]))
                if max_interval == 2:
                    # 当分箱数为2时，必然单调
                    break
                monotone = BadRateMonotone(allData, col + '_Bin', 'target')
            newVar = col + '_Bin'
            allData[newVar] = allData[col].map(lambda x:AssignBin(x, cutOff, special_attribute=[]))
            var_bin_list.append(newVar)
        else:
            max_interval=5
            # 如果有－1，则除去－1后，其他取值参与分箱
            cutOff = ChiMerge(allData, col, 'target', max_interval=max_interval, special_attribute=[-1], minBinPcnt=0)
            allData[col+'_Bin'] = allData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
            monotone = BadRateMonotone(allData, col + '_Bin', 'target')
            while (not monotone):
                max_interval -= 1
                cutOff = ChiMerge(allData, col, 'target', max_interval=max_interval, special_attribute=[-1], minBinPcnt=0)
                allData[col + '_Bin'] = allData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
                if max_interval == 3:
                    break
                monotone = BadRateMonotone(allData, col+'_Bin', 'target', ['Bin -1'])
            newVar = col + '_Bin'
            allData[newVar] = allData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
            var_bin_list.append(newVar)
        continous_merged_dict[col] = cutOff
        # 需要保存每个变量的分割点
        # save_variable_cutOffPoint = open('variable_cutOffPoint_dict.pkl','wb')
        # pickle.dump(continous_merged_dict, save_variable_cutOffPoint)
        # save_variable_cutOffPoint.close()

    '''
        第四步：WOE编码、计算IV
    '''
    WOE_dict = {}
    IV_dict = {}
    # 分箱后的变量进行编码，包括：
    # 1，初始取值个数小于5，且不需要合并的类别型变量。存放在less_value_features中
    # 2，初始取值个数小于5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 3，初始取值个数超过5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 4，连续变量。分箱后新的变量存放在var_bin_list中
    all_var = var_bin_list + less_value_feature
    for var in all_var:
        woe_iv = CalcWOE(allData, var, 'target')
        WOE_dict[var] = woe_iv['WOE']
        IV_dict[var] = woe_iv['IV']
    IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)

    IV_values = [i[1] for i in IV_dict_sorted]
    IV_name = [i[0] for i in IV_dict_sorted]
    plt.figure(figsize=(10, 10))
    plt.bar(x=range(len(IV_name)), height=IV_values, label='feature IV', alpha=0.8)
    plt.xticks(np.arange(len(IV_name)), IV_name)
    plt.xticks(rotation='90')  # x轴坐标字体旋转
    plt.subplots_adjust(left=0.25, wspace=0.25, hspace=0.25, bottom=0.4, top=0.91)  # 防止x轴字过长超出边界设置
    plt.show()

    '''
    第五步：单变量分析和多变量分析，均基于WOE编码后的值。
    （1）选择IV高于0.01的变量
    （2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
    '''
    high_IV = {k: v for k, v in IV_dict.items() if v > 0.01}
    high_IV_sorted = sorted(high_IV.items(), key=lambda x: x[1], reverse=True)
    short_list = high_IV.keys()
    short_list_2 = []
    for var in short_list:
        newVar = var + '_WOE'
        allData[newVar] = allData[var].map(WOE_dict[var])
        short_list_2.append(newVar)
    import seaborn as sns
    allDataWOE = allData[short_list_2]
    corr = allDataWOE.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(corr, cmap="YlGnBu", linewidths=0.5, vmax=1, vmin=0, annot=True, annot_kws={'size': 6, 'weight': 'bold'})
    # 热力图参数设置（相关系数矩阵，颜色，每个值间隔等）
    plt.xticks(np.arange(len(short_list_2)) + 0.5, short_list_2, fontsize=12)  # 横坐标标注点
    plt.yticks(np.arange(len(short_list_2)) + 0.5, short_list_2, fontsize=12)  # 纵坐标标注点
    ax.set_title('多变量相关性分析')  # 标题设置
    plt.subplots_adjust(left=0.25, wspace=0.25, hspace=0.25, bottom=0.4, top=0.9)
    plt.show()

    # 两两间的线性相关性检验
    # 1，将候选变量按照IV进行降序排列
    # 2，计算第i和第i+1的变量的线性相关系数
    # 3，对于系数超过阈值的两个变量，剔除IV较低的一个
    deleted_index = []
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0] + '_WOE'
        for j in range(cnt_vars):
            if i==j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0] + '_WOE'
            roh = np.corrcoef(allData[x1], allData[y1])[0, 1]
            if abs(roh) > 0.7:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)

    '''
    多变量分析：VIF
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    multi_analysis_vars_1 = [high_IV_sorted[i][0] + '_WOE' for i in range(cnt_vars) if i not in deleted_index]
    X = np.matrix(allData[multi_analysis_vars_1])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    max_VIF = max(VIF_list)
    print('最大的VIF是:{}'.format(max_VIF))
    multi_analysis_vars = multi_analysis_vars_1

    '''
    第六步：逻辑回归模型。
    要求：
    1，变量显著
    2，符号为负
    '''
    from sklearn.model_selection import train_test_split
    trainData, testData = train_test_split(allData, test_size=1/4, random_state=3)
    y_train = trainData['target']
    X_train = trainData[multi_analysis_vars]
    y_test = testData['target']
    X_test = testData[multi_analysis_vars]
    # X_train['intercept'] = [1] * X.shape[0]
    import statsmodels.api as sm
    LR = sm.Logit(y_train, X_train).fit()
    print('----'*30)
    summary = LR.summary()
    print(summary)
    pvals = LR.pvalues
    pvals = pvals.to_dict()

    train_pred = LR.predict(X_train)
    trainData['train_pred'] = train_pred
    train_ks = KS(trainData, 'train_pred', 'target')

    test_pred = LR.predict(X_test)
    testData['test_pred'] = test_pred
    test_ks = KS(testData, 'test_pred','target')

    basePoint = 600
    PDO = 20
    trainData['score'] = trainData['train_pred'].map(lambda x:Prob2Score(x, basePoint=basePoint, PDO=PDO))
    # plt.hist(trainData['score'], 100)
    # plt.xlabel('score')
    # plt.ylabel('freq')
    # plt.title('train_distribution')
    testData['score'] = testData['test_pred'].map(lambda x: Prob2Score(x, basePoint=basePoint, PDO=PDO))
    # plt.hist(testData['score'], 100)
    # plt.xlabel('score')
    # plt.ylabel('freq')
    # plt.title('test_distribution')

    # 计算基尼指数
    gini = Gini(testData,'score','target')

    # 计算LIFT提升度
    lift = lift_chart(testData)

    lorenz = draw_lorenz(testData)


    from sklearn.metrics import roc_curve, auc
    ps = psi(trainData, testData)
    fpr, tpr, threshold = roc_curve(y_test, test_pred)
    AUC = auc(fpr, tpr)
    train_ks = max(abs(tpr-fpr))


    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    train_pred2= lr.predict_proba(X_train)[:, 1]
    trainData['train_pred2'] = train_pred2
    fpr, tpr, threshold = roc_curve(y_train, train_pred2)
    AUC2 = auc(fpr, tpr)
    train_ks2 = max(abs(tpr - fpr))


if __name__ == '__main__':
    main()
