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
        :param df:
        :param target:
        :return: 删除后的数据集
    """
    df2 = df.copy()
    missing_df = missing_cal(df)
    missing_col_num = missing_df[missing_df.missing_pct >= threshold].shape[0]
    missing_col = list(missing_df[missing_df.missing_pct >= threshold].col)
    df2 = df2.drop(missing_col, axis=1)
    return df2






def data_processing(df, target):
    """
        :param df: 包含了label（target）和特征的宽表
        :param target: label（target）
        :return: 清洗后的数据集
    """
    # 特征缺失处理
    df = missing_delete_var(df, target)



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
        # df, miss_df = data_p




def main():

    df = pd.read_csv('gm_model.csv')

    df_feature = df.drop(['id_card_no', 'card_name', 'loan_date'], axis=1)

    miss = missing_cal(df_feature)
    # result_bin = ge

if __name__ == '__main__':
    main()
