# _*_ coding: utf-8 _*_

import sys
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

pd.set_option('display.float_format', lambda x: '%.3f' %x)


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
    cate_miss_col2 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in cate_col]
    num_miss_col1 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in num_col]
    num_miss_col2 = [x for x in list(miss_df[miss_df.missing_pct > 0.05]['col']) if x in num_col]
    for col in cate_miss_col1:
        df[col] = df[col].fillan('未知')
    for col in cate_miss_col2:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in num_miss_col1:
        df[col] = df[col].fillna(-999)
    for col in num_miss_col2:
        df[col] = df[col].fillna(df[col].median())

    return df, miss_df


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
        return df, miss_df


def main():

    df = pd.read_csv('gm_model.csv')

    df_feature = df.drop(['id_card_no', 'card_name', 'loan_date'], axis=1)

    # result_bin = ge
    # col_list = [x for x in df_feature.columns if x != 'label']
    # df = const_delete(df_feature, col_list, threshold=0.9)
    # desc = df.describe().T
    # miss_df = missing_cal(df_feature)
    # cate_col = list(df_feature.select_dtypes(include=['0']).columns)

    df2, miss_df = get_feature_result(df_feature, 'label')
    print(df2)
    print(miss_df)


if __name__ == '__main__':
    main()
