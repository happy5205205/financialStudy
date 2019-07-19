import sys
import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm

pd.set_option('display.float_format', lambda x: '%.3f' % x)

"""
    1.自定义缺失值处理函数
		1.1 缺失值计算

"""


def missing_cal(df):
    """
		计算特征数据缺失占比
        :param df: 数据集
        :return: 每个变量的缺失率
    """
    missing_series = df.isnull().sum() / df.shape[0]  # 此处需要修改
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index': 'col',
                                            0: 'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct', ascending=False).reset_index(drop=True)
    return missing_df


"""
	1.2 按特征（列）删除
		若字段数据缺失严重，可先检查字段特性，是业务层面设计需求，或者是数据抓取异常
		如无上述问题，建议删除缺失值占比大于设定阈值的字段
		常见阈值为90%以上或者40%~50%以上，根据特征是否对应明确的业务含义而决定是否保留
"""


def missing_delete_var(df, threshold=None):
    """
    :param df: 数据集
    :param threshold: 确实率删除的阈值
    :return: 删除后的数据集
    """
    df2 = df.copy()
    missing_df = missing_cal(df)
    missing_col_num = missing_df[missing_df.missing_pct >= threshold].shape[0]
    missing_col = list(missing_df[missing_df.missing_pct >= threshold].col)
    df2 = df2.drop(missing_col, axis=1)
    return df2


def missing_delete_user(df, threshold=None):
    df2 = df.copy()
    missing_series = df.isnull().sum(axis=1)
    missing_list = list(missing_series)
    missing_index_list = []
    for i, j in enumerate(missing_list):
        if j >= threshold:
            missing_index_list.append(i)
    df2 = df2[~(df2.index.isin(missing_index_list))]
    return df2


"""
	2. 自定义常变量处理函数
		同值化较严重的字段，如无特殊业务含义，某一数据占比超过阈值时，建议删除
"""


def const_delete(df, col_list, threshold=None):
    """

    :param df: 数据集
    :param col_list: 变量list集合
    :param threshold: 同值化处理的阈值
    :return: 处理后的数据
    """
    df2 = df.copy()
    const_col = []
    for col in col_list:
        const_pct = df2[col].value_counts().iloc[0] / df2[df2[col].notnull()].shape[0]
        if const_pct >= threshold:
            const_col.append(col)
    df2 = df2.drop(const_col, axis=1)
    return df2


"""
    3. 自定义data_processing函数，执行完整数据预处理步骤：
        1、导入数据
        2、删除缺失值（自定义函数）
        3、删除常变量（自定义函数）
          1）常变量（自定义函数）
          2）方差为0
    4、缺失值填充
          1）分类型特征填充（自定义函数）
          2）连续型特征填充（自定义函数）
"""


def data_processing(df, target):
    """
    :param df: 包含了label（target）和特征的宽表
    :param target: label（target）
    :return: 清洗后的数据集
    """
    # 特征缺失处理
    df = missing_delete_var(df, threshold=0.8)
    # 样本缺失处理
    df = missing_delete_user(df, threshold=int(df.shape[1] * 0.8))
    col_list = [x for x in df.columns if x != target]
    # 常变量处理
    df = const_delete(df, col_list, threshold=0.9)
    desc = df.describe().T
    # 剔除方差为0的特征
    std_0_col = list(desc[desc['std'] == 0].index)
    if len(std_0_col) > 0:
        df = df.drop(std_0_col, axis=1)
    df.reset_index(drop=True, inplace=True)

    # 缺失值计算和填充
    miss_df = missing_cal(df)
    # 取出类别型特征
    cate_col = list(df.select_dtypes(include=['O']).columns)
    # 取出数字型特征
    num_col = [x for x in list(df.select_dtypes(include=['int64', 'float64']).columns) if x != 'label']


    # 类别型特征缺失值大于5%以上特征
    cate_miss_col1 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in cate_col]
    # 类别型特征缺失值小于等于5%的特征
    cate_miss_col2 = [x for x in list(miss_df[miss_df.missing_pct <= 0.05]['col']) if x in cate_col]

    # 数字型特征缺失值大于5%特征
    num_miss_col1 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in num_col]
    # 数字型特征缺失值小于等于5%的特征
    num_miss_col2 = [x for x in list(miss_df[miss_df.missing_pct <= 0.05]['col']) if x in num_col]

    # 分类型特征填充
    for col in cate_miss_col1:
        df[col] = df[col].fillna('未知')
    for col in cate_miss_col2:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 数字型特征填充
    for col in num_miss_col1:
        df[col] = df[col].fillna(-999)
    for col in num_miss_col2:
        df[col] = df[col].fillna(df[col].median())

    return df, miss_df


###############################################################################################################
"""
	三、特征分箱
		分箱逻辑：
			1、类别型特征
			  1）类别数在5个以下，可以直接根据类别来分箱 (binning_cate)
			  2）类别数在5个以上，建议做降基处理，再根据降基后的类别做分箱
			2、数值型特征
			  1）离散型数值特征（特征value的变动幅度较小）：
			    若特征value的非重复计数在5个以下，可以直接根据非重复计数值来分箱(binning_cate)
			    若特征value的非重复计数在5个以上，建议根据业务解释或者数据分布做自定义分箱(binning_self)
			  2）连续型数值特征（特征value的变动幅度较大）：
			    可以用卡方分箱或自定义分箱。(binning_num,binning_self)
			    PS:一些特征用卡方分可能会报错，建议这些特征改为手动自定义分箱
			3、特征有缺失
			  1）缺失率在5%以下，可以先对缺失做填充处理再分箱(binning_num)
			  2）缺失率在5%以上，建议将缺失当作一个类别来分箱(binning_sparse_col)
			4、稀疏特征分箱
			  建议将稀疏值（一般为0）单独分为一箱，剩下的值做卡方或者自定义分箱(binning_sparse_col)
"""

"""
	1.自定义指标评估函数
		KS、precision、 tpr、 fpr
"""


def cal_ks(df, col, target):
    """

    :param df: 数据集
    :param col:输入特征
    :param target:好坏标记的字段名
    :return: KS值, precision准确率, tpr召回率, fpr打扰率
    """

    bad = df[target].sum()
    good = df[target].count() - bad
    value_list = list(df[col])
    label_list = list(df[target])
    value_count = df[col].nunique()

    items = sorted(zip(value_list, label_list), key=lambda x: x[0])

    value_bin = []
    ks_list = []
    if value_count <= 200:
        for i in sorted(set(value_list)):
            value_bin.append(i)
            label_bin = [x[1] for x in items if x[0] < i]
            badrate = sum(label_bin) / bad
            goodrate = (len(label_bin) - sum(label_bin)) / good
            ks = abs(goodrate - badrate)
            ks_list.append(ks)
    else:
        for i in range(1, 201):
            step = (max(value_list) - min(value_list)) / 200
            idx = min(value_list) + i * step
            value_bin.append(idx)
            label_bin = [x[1] for x in items if x[0] < idx]
            badrate = sum(label_bin) / bad
            goodrate = (len(label_bin) - sum(label_bin)) / good
            ks = abs(goodrate - badrate)
            ks_list.append(ks)
    ks = round(max(ks_list), 3)

    ks_value = [value_bin[i] for i, j in enumerate(ks_list) if j == max(ks_list)][0]
    precision = df[(df[col] <= ks_value) & (df[target] == 1)].shape[0] / df[df[col] <= ks_value].shape[0]
    tpr = df[(df[col] <= ks_value) & (df[target] == 1)].shape[0] / bad
    fpr = df[(df[col] <= ks_value) & (df[target] == 0)].shape[0] / good

    return ks, precision, tpr, fpr


"""
	2.自定义卡方分箱函数
		2.1 变量分割点
"""


def split_data(df, col, split_num):
    """
	先用卡方分箱输出变量的分割点
    :param df: 原始数据
    :param col: 需要分箱的变量
    :param split_num: 分割点的数量
    :return:
    """
    df2 = df.copy()
    count = df2.shape[0]  # 总样本数
    n = math.floor(count / split_num)  # 按照分割点数目等分后每组的样本数
    split_index = [i * n for i in range(1, split_num)]  # 分割点的索引
    values = sorted(list(df2[col]))  # 对变量的值从小到大进行排序
    split_value = [values[i] for i in split_index]  # 分割点对应的value
    split_value = sorted(list(set(split_value)))  # 分割点的value去重排序
    return split_value


def assign_group(x, split_bin):
    n = len(split_bin)
    if x <= min(split_bin):
        return min(split_bin)  # 如果x小于分割点的最小值，则x映射为分割点的最小值
    elif x > max(split_bin):  # 如果x大于分割点的最大值，则x映射为分割点的最大值
        return 10e10
    else:
        for i in range(n - 1):
            if split_bin[i] < x <= split_bin[i + 1]:  # 如果x在两个分割点之间，则x映射为分割点较大的值
                return split_bin[i + 1]


"""
	2.2 计算违约率
"""


def bin_bad_rate(df, col, target, grantRateIndicator=0):
    """

    :param df: 原始数据
    :param col: 原始变量/变量映射后的字段
    :param target:目标变量的字段
    :param grantRateIndicator:是否输出整体违约率
    :return:
    """
    total = df.groupby([col])[target].count()
    bad = df.groupby([col])[target].sum()
    total_df = pd.DataFrame({'total': total})
    bad_df = pd.DataFrame({'bad': bad})
    regroup = pd.merge(total_df, bad_df, left_index=True, right_index=True, how='left')
    regroup = regroup.reset_index()
    regroup['bad_rate'] = regroup['bad'] / regroup['total']  # 计算根据col分组后每组的违约率
    dict_bad = dict(zip(regroup[col], regroup['bad_rate']))  # 转为字典形式
    if grantRateIndicator == 0:
        return (dict_bad, regroup)
    total_all = df.shape[0]
    bad_all = df[target].sum()
    all_bad_rate = bad_all / total_all  # 计算总体的违约率
    return (dict_bad, regroup, all_bad_rate)


"""
	2.3 计算卡方值
"""


def cal_chi2(df, all_bad_rate):
    df2 = df.copy()
    df2['expected'] = df2['total'] * all_bad_rate  # 计算每组的坏用户期望数量
    combined = zip(df2['expected'], df2['bad'])  # 遍历每组的坏用户期望数量和实际数量
    chi = [(i[0] - i[1]) ** 2 / i[0] for i in combined]  # 计算每组的卡方值
    chi2 = sum(chi)  # 计算总的卡方值
    return chi2


def assign_bin(x, cutoffpoints):
    bin_num = len(cutoffpoints) + 1  # 箱体个数
    if x <= cutoffpoints[0]:  # 如果x小于最小的cutoff点，则映射为Bin 0
        return 'Bin 0'
    elif x > cutoffpoints[-1]:  # 如果x大于最大的cutoff点，则映射为Bin(bin_num-1)
        return 'Bin {}'.format(bin_num - 1)
    else:
        for i in range(0, bin_num - 1):
            if cutoffpoints[i] < x <= cutoffpoints[i + 1]:  # 如果x在两个cutoff点之间，则x映射为Bin(i+1)
                return 'Bin {}'.format(i + 1)

"""
	2.4 卡方分箱（干货）
"""


def ChiMerge(df, col, target, max_bin=5, min_binpct=0):
    col_unique = sorted(list(set(df[col])))  # 变量的唯一值并排序
    n = len(col_unique)  # 变量唯一值得个数
    df2 = df.copy()
    if n > 100:  # 如果变量的唯一值数目超过100，则将通过split_data和assign_group将x映射为split对应的value
        split_col = split_data(df2, col, 100)  # 通过这个目的将变量的唯一值数目人为设定为100
        df2['col_map'] = df2[col].map(lambda x: assign_group(x, split_col))
    else:
        df2['col_map'] = df2[col]  # 变量的唯一值数目没有超过100，则不用做映射
    # 生成dict_bad,regroup,all_bad_rate的元组
    (dict_bad, regroup, all_bad_rate) = bin_bad_rate(df2, 'col_map', target, grantRateIndicator=1)
    col_map_unique = sorted(list(set(df2['col_map'])))  # 对变量映射后的value进行去重排序
    group_interval = [[i] for i in col_map_unique]  # 对col_map_unique中每个值创建list并存储在group_interval中

    while (len(group_interval) > max_bin):  # 当group_interval的长度大于max_bin时，执行while循环
        chi_list = []
        for i in range(len(group_interval) - 1):
            temp_group = group_interval[i] + group_interval[i + 1]  # temp_group 为生成的区间,list形式，例如[1,3]
            chi_df = regroup[regroup['col_map'].isin(temp_group)]
            chi_value = cal_chi2(chi_df, all_bad_rate)  # 计算每一对相邻区间的卡方值
            chi_list.append(chi_value)
        best_combined = chi_list.index(min(chi_list))  # 最小的卡方值的索引
        # 将卡方值最小的一对区间进行合并
        group_interval[best_combined] = group_interval[best_combined] + group_interval[best_combined + 1]
        # 删除合并前的右区间
        group_interval.remove(group_interval[best_combined + 1])
        # 对合并后每个区间进行排序
    group_interval = [sorted(i) for i in group_interval]
    # cutoff点为每个区间的最大值
    cutoffpoints = [max(i) for i in group_interval[:-1]]

    # 检查是否有箱只有好样本或者只有坏样本
    df2['col_map_bin'] = df2['col_map'].apply(lambda x: assign_bin(x, cutoffpoints))  # 将col_map映射为对应的区间Bin
    # 计算每个区间的违约率
    (dict_bad, regroup) = bin_bad_rate(df2, 'col_map_bin', target)
    # 计算最小和最大的违约率
    [min_bad_rate, max_bad_rate] = [min(dict_bad.values()), max(dict_bad.values())]
    # 当最小的违约率等于0，说明区间内只有好样本，当最大的违约率等于1，说明区间内只有坏样本
    while min_bad_rate == 0 or max_bad_rate == 1:
        bad01_index = regroup[regroup['bad_rate'].isin([0, 1])].col_map_bin.tolist()  # 违约率为1或0的区间
        bad01_bin = bad01_index[0]
        if bad01_bin == max(regroup.col_map_bin):
            cutoffpoints = cutoffpoints[:-1]  # 当bad01_bin是最大的区间时，删除最大的cutoff点
        elif bad01_bin == min(regroup.col_map_bin):
            cutoffpoints = cutoffpoints[1:]  # 当bad01_bin是最小的区间时，删除最小的cutoff点
        else:
            bad01_bin_index = list(regroup.col_map_bin).index(bad01_bin)  # 找出bad01_bin的索引
            prev_bin = list(regroup.col_map_bin)[bad01_bin_index - 1]  # bad01_bin前一个区间
            df3 = df2[df2.col_map_bin.isin([prev_bin, bad01_bin])]
            (dict_bad, regroup1) = bin_bad_rate(df3, 'col_map_bin', target)
            chi1 = cal_chi2(regroup1, all_bad_rate)  # 计算前一个区间和bad01_bin的卡方值
            later_bin = list(regroup.col_map_bin)[bad01_bin_index + 1]  # bin01_bin的后一个区间
            df4 = df2[df2.col_map_bin.isin([later_bin, bad01_bin])]
            (dict_bad, regroup2) = bin_bad_rate(df4, 'col_map_bin', target)
            chi2 = cal_chi2(regroup2, all_bad_rate)  # 计算后一个区间和bad01_bin的卡方值
            if chi1 < chi2:  # 当chi1<chi2时,删除前一个区间对应的cutoff点
                cutoffpoints.remove(cutoffpoints[bad01_bin_index - 1])
            else:  # 当chi1>=chi2时,删除bin01对应的cutoff点
                cutoffpoints.remove(cutoffpoints[bad01_bin_index])
        df2['col_map_bin'] = df2['col_map'].apply(lambda x: assign_bin(x, cutoffpoints))
        (dict_bad, regroup) = bin_bad_rate(df2, 'col_map_bin', target)
        # 重新将col_map映射至区间，并计算最小和最大的违约率，直达不再出现违约率为0或1的情况，循环停止
        [min_bad_rate, max_bad_rate] = [min(dict_bad.values()), max(dict_bad.values())]

    # 检查分箱后的最小占比
    if min_binpct > 0:
        group_values = df2['col_map'].apply(lambda x: assign_bin(x, cutoffpoints))
        df2['col_map_bin'] = group_values  # 将col_map映射为对应的区间Bin
        group_df = group_values.value_counts().to_frame()
        group_df['bin_pct'] = group_df['col_map'] / sum(group_df['col_map'])  # 计算每个区间的占比
        min_pct = group_df.bin_pct.min()  # 得出最小的区间占比
        while min_pct < min_binpct and len(cutoffpoints) > 2:  # 当最小的区间占比小于min_pct且cutoff点的个数大于2，执行循环
            # 下面的逻辑基本与“检验是否有箱体只有好/坏样本”的一致
            min_pct_index = group_df[group_df.bin_pct == min_pct].index.tolist()
            min_pct_bin = min_pct_index[0]
            if min_pct_bin == max(group_df.index):
                cutoffpoints = cutoffpoints[:-1]
            elif min_pct_bin == min(group_df.index):
                cutoffpoints = cutoffpoints[1:]
            else:
                minpct_bin_index = list(group_df.index).index(min_pct_bin)
                prev_pct_bin = list(group_df.index)[minpct_bin_index - 1]
                df5 = df2[df2['col_map_bin'].isin([min_pct_bin, prev_pct_bin])]
                (dict_bad, regroup3) = bin_bad_rate(df5, 'col_map_bin', target)
                chi3 = cal_chi2(regroup3, all_bad_rate)
                later_pct_bin = list(group_df.index)[minpct_bin_index + 1]
                df6 = df2[df2['col_map_bin'].isin([min_pct_bin, later_pct_bin])]
                (dict_bad, regroup4) = bin_bad_rate(df6, 'col_map_bin', target)
                chi4 = cal_chi2(regroup4, all_bad_rate)
                if chi3 < chi4:
                    cutoffpoints.remove(cutoffpoints[minpct_bin_index - 1])
                else:
                    cutoffpoints.remove(cutoffpoints[minpct_bin_index])
            group_values = df2['col_map'].apply(lambda x: assign_bin(x, cutoffpoints))
            df2['col_map_bin'] = group_values  # 将col_map映射为对应的区间Bin
            group_df = group_values.value_counts().to_frame()
            group_df['bin_pct'] = group_df['col_map'] / sum(group_df['col_map'])  # 计算每个区间的占比
            min_pct = group_df.bin_pct.min()  # 得出最小的区间占比
    return cutoffpoints


"""
	3. 自定义变量分箱函数
		3.1 类别型特征
"""


def binning_cate(df, col, target):
    """

    :param df: 数据集
    :param col: 输入特征
    :param target: 好坏标记的字段名
    :return: bin_df 特征的评估结果
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    d1 = df.groupby([col], as_index=True)
    d2 = pd.DataFrame()
    d2['样本数'] = d1[target].count()
    d2['黑样本数'] = d1[target].sum()
    d2['白样本数'] = d2['样本数'] - d2['黑样本数']
    d2['逾期用户占比'] = d2['黑样本数'] / d2['样本数']
    d2['badattr'] = d2['黑样本数'] / bad
    d2['goodattr'] = d2['白样本数'] / good
    d2['WOE'] = np.log(d2['badattr'] / d2['goodattr'])
    d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['WOE']
    d2['IV'] = d2['bin_iv'].sum()

    bin_df = d2.reset_index()
    bin_df.drop(['badattr', 'goodattr', 'bin_iv'], axis=1, inplace=True)
    bin_df.rename(columns={col: '分箱结果'}, inplace=True)
    bin_df['特征名'] = col
    bin_df = pd.concat([bin_df['特征名'], bin_df.iloc[:, :-1]], axis=1)
    return bin_df


"""
	3.2 数值型特征
		3.2.1 离散型数值特征
"""


def binning_self(df, col, target, cut=None, right_border=True):
    """
    :param df: 数据集
    :param col: 输入的特征
    :param target: 好坏标记的字段名
    :param cut: 总定义划分区间的list
    :param right_border: 设置左开右闭，左闭右开
    :return: bin_df 特征的评估结果
    """

    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    bucket = pd.cut(df[col], cut, right=right_border)
    d1 = df.groupby(bucket)
    d2 = pd.DataFrame()
    d2['样本数'] = d1[target].count()
    d2['黑样本数'] = d1[target].sum()
    d2['白样本数'] = d2['样本数'] - d2['黑样本数']
    d2['逾期用户占比'] = d2['黑样本数'] / d2['样本数']
    d2['badattr'] = d2['黑样本数'] / bad
    d2['goodattr'] = d2['白样本数'] / good
    d2['WOE'] = np.log(d2['badattr'] / d2['goodattr'])
    d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['WOE']
    d2['IV'] = d2['bin_iv'].sum()

    bin_df = d2.reset_index()
    bin_df.drop(['badattr', 'goodattr', 'bin_iv'], axis=1, inplace=True)
    bin_df.rename(columns={col: '分箱结果'}, inplace=True)
    bin_df['特征名'] = col
    bin_df = pd.concat([bin_df['特征名'], bin_df.iloc[:, :-1]], axis=1)

    ks, precision, tpr, fpr = cal_ks(df, col, target)
    bin_df['准确率'] = precision
    bin_df['召回率'] = tpr
    bin_df['打扰率'] = fpr
    bin_df['KS'] = ks

    return bin_df


"""
	3.2.2 连续型数值特征
"""


def binning_num(df, target, col, max_bin=None, min_binpct=None):
    """

    :param df: 数据集
    :param target: 好坏标记的字段名
    :param col: 输入的特征
    :param max_bin: 最大分箱个数
    :param min_binpct: 区间内样本所占总体的最小比
    :return:
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    inf = float('inf')
    ninf = float('-inf')

    cut = ChiMerge(df, col, target, max_bin=max_bin, min_binpct=min_binpct)
    cut.insert(0, ninf)
    cut.append(inf)
    bucket = pd.cut(df[col], cut)
    d1 = df.groupby(bucket)
    d2 = pd.DataFrame()
    d2['样本数'] = d1[target].count()
    d2['黑样本数'] = d1[target].sum()
    d2['白样本数'] = d2['样本数'] - d2['黑样本数']
    d2['逾期用户占比'] = d2['黑样本数'] / d2['样本数']
    d2['badattr'] = d2['黑样本数'] / bad
    d2['goodattr'] = d2['白样本数'] / good
    d2['WOE'] = np.log(d2['badattr'] / d2['goodattr'])
    d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['WOE']
    d2['IV'] = d2['bin_iv'].sum()

    bin_df = d2.reset_index()
    bin_df.drop(['badattr', 'goodattr', 'bin_iv'], axis=1, inplace=True)
    bin_df.rename(columns={col: '分箱结果'}, inplace=True)
    bin_df['特征名'] = col
    bin_df = pd.concat([bin_df['特征名'], bin_df.iloc[:, :-1]], axis=1)

    ks, precision, tpr, fpr = cal_ks(df, col, target)
    bin_df['准确率'] = precision
    bin_df['召回率'] = tpr
    bin_df['打扰率'] = fpr
    bin_df['KS'] = ks

    return bin_df


"""
	3.3 特征分箱
"""


def binning_sparse_col(df, target, col, max_bin=None, min_binpct=0, sparse_value=None):
    """
    :param df: 数据集
    :param target: 好坏标记的字段名
    :param col: 输入的特征
    :param max_bin: 最大分箱个数
    :param min_binpct: 区间内样本所占总体的最小比
    :param sparse_value: 单独分为一箱的values值
    :return: 特征的评估结果
    """

    total = df[target].count()
    bad = df[target].sum()
    good = total - bad

    # 对稀疏值0值或者缺失值单独分箱
    temp1 = df[df[col] == sparse_value]
    temp2 = df[~(df[col] == sparse_value)]

    bucket_sparse = pd.cut(temp1[col], [float('-inf'), sparse_value])
    group1 = temp1.groupby(bucket_sparse)
    bin_df1 = pd.DataFrame()
    bin_df1['样本数'] = group1[target].count()
    bin_df1['黑样本数'] = group1[target].sum()
    bin_df1['白样本数'] = bin_df1['样本数'] - bin_df1['黑样本数']
    bin_df1['逾期用户占比'] = bin_df1['黑样本数'] / bin_df1['样本数']
    bin_df1['badattr'] = bin_df1['黑样本数'] / bad
    bin_df1['goodattr'] = bin_df1['白样本数'] / good
    bin_df1['WOE'] = np.log(bin_df1['badattr'] / bin_df1['goodattr'])
    bin_df1['bin_iv'] = (bin_df1['badattr'] - bin_df1['goodattr']) * bin_df1['WOE']

    bin_df1 = bin_df1.reset_index()

    # 对剩余部分做卡方分箱
    cut = ChiMerge(temp2, col, target, max_bin=max_bin, min_binpct=min_binpct)
    cut.insert(0, sparse_value)
    cut.append(float('inf'))

    bucket = pd.cut(temp2[col], cut)
    group2 = temp2.groupby(bucket)
    bin_df2 = pd.DataFrame()
    bin_df2['样本数'] = group2[target].count()
    bin_df2['黑样本数'] = group2[target].sum()
    bin_df2['白样本数'] = bin_df2['样本数'] - bin_df2['黑样本数']
    bin_df2['逾期用户占比'] = bin_df2['黑样本数'] / bin_df2['样本数']
    bin_df2['badattr'] = bin_df2['黑样本数'] / bad
    bin_df2['goodattr'] = bin_df2['白样本数'] / good
    bin_df2['WOE'] = np.log(bin_df2['badattr'] / bin_df2['goodattr'])
    bin_df2['bin_iv'] = (bin_df2['badattr'] - bin_df2['goodattr']) * bin_df2['WOE']

    bin_df2 = bin_df2.reset_index()

    # 合并分箱结果
    bin_df = pd.concat([bin_df1, bin_df2], axis=0)
    bin_df['IV'] = bin_df['bin_iv'].sum().round(3)

    bin_df.drop(['badattr', 'goodattr', 'bin_iv'], axis=1, inplace=True)
    bin_df.rename(columns={col: '分箱结果'}, inplace=True)
    bin_df['特征名'] = col
    bin_df = pd.concat([bin_df['特征名'], bin_df.iloc[:, :-1]], axis=1)

    ks, precision, tpr, fpr = cal_ks(df, col, target)
    bin_df['准确率'] = precision
    bin_df['召回率'] = tpr
    bin_df['打扰率'] = fpr
    bin_df['KS'] = ks

    return bin_df


"""
	四. 自定义get_feature_result函数，执行完整数据预处理步骤：
		1、数据预处理，调用data_processing函数
		2、变量分箱
		  1）类别型变量分箱
		  2）数值型变量分箱
		  2）卡方分箱报错的变量分箱
		3、得到分箱结果feature_result及其评估指标
		   order_col = ['特征名', '分箱结果', '样本数', '黑样本数', '白样本数', '逾期用户占比', 'WOE', 'IV', '准确率', '召回率', '打扰率', 'KS']
"""


def get_feature_result(df, target):
    """

    :param df: 含有特征和标签的宽表
    :param target: 好坏标签字段名
    :return: 每个特征的评估结果
    """
    if target not in df.columns:

        print('请将特征文件关联样本好坏标签(字段名label)后再重新运行!')

    else:

        print('----------------------------------数据清洗开始----------------------------------')
        df, miss_df = data_processing(df, target)
        print('----------------------------------数据清洗完成----------------------------------')

        cate_col = list(df.select_dtypes(include=['O']).columns)
        num_col = [x for x in list(df.select_dtypes(include=['int64', 'float64']).columns) if x != 'label']

        print('----------------------------------数据分箱开始----------------------------------')
        # 类别型变量分箱
        print('类别型变量分箱开始')
        bin_cate_list = []
        for col in tqdm(cate_col):
            bin_cate = binning_cate(df, col, target)
            bin_cate['rank'] = list(range(1, bin_cate.shape[0] + 1, 1))
            bin_cate_list.append(bin_cate)
        print('类别型变量分箱结束')

        # 数值型特征分箱
        num_col1 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in num_col]
        num_col2 = [x for x in list(miss_df[miss_df.missing_pct <= 0.05]['col']) if x in num_col]

        print('数值型变量分箱开始')
        bin_num_list1 = []
        err_col1 = []
        for col in tqdm(num_col1):
            try:
                bin_df1 = binning_sparse_col(df, 'label', col, min_binpct=0.05, max_bin=4, sparse_value=-99998)
                bin_df1['rank'] = list(range(1, bin_df1.shape[0] + 1, 1))
                bin_num_list1.append(bin_df1)
            except (IndexError, ZeroDivisionError):
                err_col1.append(col)
            continue

        bin_num_list2 = []
        err_col2 = []
        for col in tqdm(num_col2):
            try:
                bin_df2 = binning_num(df, 'label', col, min_binpct=0.05, max_bin=5)
                bin_df2['rank'] = list(range(1, bin_df2.shape[0] + 1, 1))
                bin_num_list2.append(bin_df2)
            except (IndexError, ZeroDivisionError):
                err_col2.append(col)
            continue
        print('数值型变量分箱结束')

        # 卡方分箱报错的特征分箱
        print('卡方分箱报错的变量分箱开始')
        err_col = err_col1 + err_col2
        bin_num_list3 = []
        if len(err_col) > 0:
            for col in tqdm(err_col):
                ninf = float('-inf')
                inf = float('inf')
                q_25 = df[col].quantile(0.25)
                q_50 = df[col].quantile(0.5)
                q_75 = df[col].quantile(0.75)

                cut = list(sorted(set([ninf, q_25, q_50, q_75, inf])))

                bin_df3 = binning_self(df, col, target, cut=cut, right_border=True)
                bin_df3['rank'] = list(range(1, bin_df3.shape[0] + 1, 1))
                bin_num_list3.append(bin_df3)
        print('卡方分箱报错的变量分箱结束')
        print('----------------------------------数据分箱结束----------------------------------')

        bin_all_list = bin_num_list1 + bin_num_list2 + bin_num_list3 + bin_cate_list

        feature_result = pd.concat(bin_all_list, axis=0)
        feature_result = feature_result.sort_values(['IV', 'rank'], ascending=[False, True])
        feature_result = feature_result.drop(['rank'], axis=1)
        order_col = ['特征名', '分箱结果', '样本数', '黑样本数', '白样本数', '逾期用户占比', 'WOE', 'IV', '准确率', '召回率', '打扰率', 'KS']
        feature_result = feature_result[order_col]
        return feature_result


"""
	五. 导入数据，运行函数，实现自动化特征评估功能
"""


def main():
    # if len(sys.argv)==1:
    #     print('请数据特征数据文件：')
    #     sys.exit()
    # feature_file = sys.argv[1]
    # file_path=os.getcwd()
    # if 'xlsx' in feature_file or 'xls' in feature_file:
    #     df = pd.read_excel(file_path+'/'+feature_file,encoding='gbk')
    # else:
    #     df = pd.read_csv(file_path+'/'+feature_file, encoding='gbk')

    df = pd.read_csv('gm_model.csv')

    df_feature = df.drop(['id_card_no', 'card_name', 'loan_date'], axis=1)
    result_bin = get_feature_result(df_feature, 'label')
    # print(result_bin)
    result_bin.to_csv('estimate_result.csv', sep=',', encoding='gbk', index=False)


if __name__ == '__main__':
    main()