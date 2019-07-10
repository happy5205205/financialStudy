# _*_ coding: utf-8 _*_

import sys
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

pd.set_option('display.float_format', lambda x: '%.3f' %x)

# ---------------------------第一部分：主要是数据清洗---------------------------


def missing_cal(df):
    """
    计算特征数据缺失占比
    :param df: 数据集
    :return: 每个变量的确实值
    """
    missing_series = df.isnull().sum() / df.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index': 'col', 0: 'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct', ascending=False).reset_index(drop=True)
    return missing_df


def missing_delete_var(df, threshold=None):
    """
        特征缺失处理
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


def missing_delete_user(df, threshold):

    """
    样本缺失处理
    :param df:
    :param threshold:
    :return:
    """
    df2 = df.copy()
    missing_series = df.isnull().sum(axis=1)
    missing_list = list(missing_series)
    missing_index_list = []
    for i, j in enumerate(missing_list):
        if j > threshold:
            missing_index_list.append(i)
    df2 = df2[~(df2.index.isin(missing_index_list))]
    return df2


def const_delete(df, col_list, threshold=None):
    """
        自定义常变量处理函数,同值化较严重的字段，如无特殊业务含义，某一数据占比超过阈值时，建议删除
        :param df: 数据集
        :param col_list: 变量list集合
        :param threshold: 同值化处理的阈值
        :return: 处理后的数据
    """
    df2 = df.copy()
    const_col = []
    for col in col_list:
        const_pct = df2[col].value_counts().iloc[0] / df2[df2[col].notnull()].shape[0]
        if const_pct > threshold:
            const_col.append(col)
    df2 = df2.drop(const_col, axis=1)
    return df2


def data_processing(df, target):
    """
        :param df: 包含了label（target）和特征的宽表
        :param target: label（target）
        :return: 清洗后的数据集
    """
    # 特征缺失处理
    df = missing_delete_var(df, threshold=0.8)
    # 样本缺失处理
    df = missing_delete_user(df, threshold=int(df.shape[1]*0.8))
    # 常变量处理
    col_list = [x for x in df.columns if x != target]
    df = const_delete(df, col_list, threshold=0.9)
    # 删除方差为0的特征
    desc = df.describe().T
    std_0_col = list(desc[desc['std'] == 0].index)
    if len(std_0_col) > 0:
        df = df.drop(std_0_col, axis=1)
    df.reset_index(drop=True, inplace=True)

    # 缺失值计算和填充
    miss_df = missing_cal(df)
    cate_col = list(df.select_dtypes(include=['O']).columns)  # O大写的O代表Object属于类别型特征
    num_col = [x for x in list(df.select_dtypes(include=['float64', 'int64']).columns) if x != 'label']

    # 分类型特征填充
    cate_miss_col1 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in cate_col]
    cate_miss_col2 = [x for x in list(miss_df[miss_df.missing_pct <= 0.05]['col']) if x in cate_col]
    num_miss_col1 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in num_col]
    num_miss_col2 = [x for x in list(miss_df[miss_df.missing_pct <= 0.05]['col']) if x in num_col]
    for col in cate_miss_col1:
        df[col] = df[col].fillan('未知')
    for col in cate_miss_col2:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in num_miss_col1:
        df[col] = df[col].fillna(-999)
    for col in num_miss_col2:
        df[col] = df[col].fillna(df[col].median())

    return df, miss_df

# ---------------------------第二部分：特征分箱---------------------------


def binning_cate(df, col, target):
    """
    类别型特征进行分箱
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
    d2['白样本数'] = d2['样本数']-d2['黑样本数']
    d2['逾期用户占比'] = d2['黑样本数'] / d2['样本数']
    d2['badattr'] = d2['黑样本数'] / bad
    d2['goodattr'] = d2['白样本数'] / good
    d2['WOE'] = np.log(d2['badattr']/d2['goodattr'])
    d2['bin_iv'] = (d2['badattr']-d2['goodattr'])*d2['WOE']
    d2['IV'] = d2['bin_iv'].sum()

    bin_df = d2.reset_index()
    bin_df.drop(['badattr', 'goodattr', 'bin_iv'], axis=1, inplace=True)
    bin_df.rename(columns={col: '分箱结果'},inplace=True)
    bin_df['特征名'] = col
    bin_df = pd.concat([bin_df['特征名'], bin_df.iloc[:, :-1]], axis=1)
    return bin_df

#---------------------卡方分箱-----------------------------
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
    n = math.floor(count / split_num) # 按照分割点数目等分后每组的样本数
    split_index = [i*n for i in range(1, split_num)]
    values = sorted(list(df2[col]))
    split_value = [values[i] for i in split_index]
    split_value = sorted(list(set(split_value)))

    return split_value


def assign_group(x, split_bin):
    n = len(split_bin)
    if x < min(split_bin):
        return min(split_bin)  # 如果x小于分割点的最小值，则x映射为分割点的最小值
    elif x > max(split_bin): # 如果x大于分割点的最大值，则x映射为分割点的最大值
        return max(split_bin)
    else:
        for i in range(n-1):
            if split_bin[i] < x < split_bin[i+1]:  # 如果x在两个分割点之间，则x映射为分割点较大的值
                return split_bin[i+1]

def bin_bad_rate(df, col, target, grantRateIndicator=0):
    """
        分组的违约概率
        :param df: 原始数据
        :param col: 原始变量/变量映射后的字段
        :param target:目标变量的字段
        :param grantRateIndicator:是否输出整体违约率
        :return:
    """
    total = df.groupby(col)[target].count()
    bad = df.groupby(col)[target].sum()
    total_df = pd.DataFrame({'total': total})
    bad_df = pd.DataFrame({'bad':bad})
    regroup = pd.merge(total_df, bad_df, how='left', left_index=True, right_index=True)
    regroup = regroup.reset_index()
    regroup['bad_rate'] = regroup['bad'] / regroup['total']
    dict_bad = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dict_bad, regroup)
    else:
        total_all = df.shape[0]
        bad_all = sum(df[target])
        all_bad_rate = total_all/bad_all
        return (dict_bad, regroup, all_bad_rate)


def ChiMerge(df, col, target, max_bin=5, min_binpct=0):
    col_unique = sorted(list(set(df[col]))) # 变量的唯一值并排序
    n = len(col_unique) # 变量唯一值的个数
    df2 = df.copy()
    if n > 100: # 如果变量的唯一值数目超过100，则将通过split_data和assign_group将x映射为split对应的value
        split_col = split_data(df2, col, 100)
        df2['col_map'] = df2[col].map(lambda x:assign_group(x, split_col))
    else:
        df2['col_map'] = df2[col] # 变量的唯一值数目没有超过100，则不用做映射
    # 生成dict_bad,regroup,all_bad_rate的元组





def binning_sparse_col(df, target, col, max_bin=None, min_binpct=None, sqarse_value=None):
    """
        缺失率大于0.05的特征分箱
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
    temp1 = df[df[col] == sqarse_value]
    temp2 = df[~df[col] == sqarse_value]

    bucket_sparse = pd.cut(temp1[col], [float('-inf'), sqarse_value])
    group1 = temp1.groupby(bucket_sparse)
    bin_df1 = pd.DataFrame()
    bin_df1['样本数'] = group1[target].count
    bin_df1['黑样本数'] = group1[target].sum()
    bin_df1['白样本数'] = bin_df1['样本数'] - bin_df1['黑样本数']
    bin_df1['逾期用户占比'] = bin_df1['黑样本数']/bin_df1['样本数']
    bin_df1['badattr'] = bin_df1['黑样本数'] /bad
    bin_df1['goodattr'] = bin_df1['白样本数'] / good
    bin_df1['WOE'] = np.log(bin_df1['badattr']/bin_df1['goodattr'])
    bin_df1['IV'] = (bin_df1['badattr'] - bin_df1['goodattr'])*bin_df1['WOE']

    bin_df1.reset_index()

    # 对剩余做卡方分箱
    # cut = ChiMerge(temp2, col, target, max_bin=max_bin, min_binpct=min_binpct)





def get_feature_result(df_feature, target):
    """
        :param df_feature: 含有特征和标签的宽表
        :param target: 好坏标签字段名
        :return: 每个特征的评估结果
    """
    if target not in df_feature.columns:
        print('请将特征文件关联 样本还坏标签（字段名label）后在重新运行')
    else:
        print('数据清洗开始')
        df, miss_df = data_processing(df_feature, target)
        print('数据清洗完成')

        cate_col = list(df.select_dtypes(include=['O']).columns)
        num_col = [x for x in list(df.select_dtypes(include=['int64', 'float64']).columns) if x != target]

        # 类别型变量分箱
        bin_cate_list = []
        print('类别型特征开始分箱')
        for col in tqdm(cate_col):
            bin_cate = binning_cate(df_feature, col, target)
            bin_cate['rank'] = list(range(1, bin_cate.shape[0] + 1, 1))
            bin_cate_list.append(bin_cate)
        print('类别型特征分箱结束')

        num_col1 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in num_col]
        num_col2 = [x for x in list(miss_df[miss_df.missing_pct <= 0.05]['col']) if x in num_col]

        print('开始特征分箱')
        bin_num_list1 = []
        err_col1 = []
        for col in tqdm(num_col1):
            try:
                # bin
                pass










def main():
    """
        主函数
    """

    df = pd.read_csv('test_file.csv')

    df_feature = df.drop(['id_card_no', 'card_name', 'loan_date'], axis=1)

    # df2, miss_df = get_feature_result(df_feature, 'label')

    get_feature_result(df_feature, 'label')


if __name__ == '__main__':
    main()
