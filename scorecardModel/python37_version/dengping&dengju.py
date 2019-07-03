# _*_ coding:utf-8 _*_
"""
    等频和等距分箱
"""
import pandas as pd
import numpy as np
import os
#from scorecardModel.python37_version import utils_v3
from sklearn.model_selection import train_test_split
import pickle
import utils_v3
import warnings

warnings.filterwarnings('ignore')

def main():

    data_path = './data'
    outPath = './newResult'
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    data = pd.read_csv(os.path.join(data_path, 'test_file.csv'))

    # 缺失值替换成-1
    miss_data = [-999977, -999976, -999978, -99998, -999979, -99998]

    for i in miss_data:
        data.replace(i, -1, inplace=True)

    more_value_feature = []  # 存放变量取值大于5
    less_value_feature = []  # 存放变量取值少于5

    # columnName = list(data.columns)
    columnNameRemove =['id_card_no', 'card_name', 'loan_date', 'target']

    for col in [cols for cols in data.columns if cols not in columnNameRemove]:
        valueCount = len(set(data[col]))
        if valueCount > 5:
            more_value_feature.append(col)
        else:
            less_value_feature.append(col)
    trainData, testData = train_test_split(data, test_size=1/4)

    merge_bin_dict = {}  # 存放需要合并的变量，以及合并方法
    var_bin_list = []  # 由于某个取值没有好或者坏样本而需要合并的变量

    for col in less_value_feature:
        binBadRate = utils_v3.BinBadRate(df=trainData, col=col, target='target')[0]
        if min(binBadRate.values()) == 0:
            print('{}标签中存在坏样本比例为0，需要合并'.format(col))
            combine_bin = utils_v3.MergeBad0(df=trainData, col=col, target='target')
            # print(combine_bin)
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            trainData[newVar] = trainData[col].map(combine_bin)
            var_bin_list.append(newVar)
        if max(binBadRate.values()) == 1:
            print('{}标签中存在好样本比例为0，需要合并'.format(col))
            combine_bin = utils_v3.MergeBad0(df=trainData, col=col, target='target')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            trainData[newVar] = trainData[col].map(combine_bin)
            var_bin_list.append(newVar)
        # 保存需要合并的变量，以及合并方法merge_bin_dict
        merge_bin_dict_file = open(outPath + 'merge_bin_dict.pkl', 'wb')
        pickle.dump(merge_bin_dict, merge_bin_dict_file)
        merge_bin_dict_file.close()

        # less_value_feature中剔除不需要合并的变量
        less_value_feature = [i for i in less_value_feature if i + '_Bin' not in var_bin_list]

    for col in more_value_feature:
        print('{} is in processing'.format(col))
        splitPoint = utils_v3.UnsupervisedSplitBin(df=trainData, var=col, numOfSplit = 5, method = 'equal freq1')
        print('变量{}的切割点为{}'.format(col, splitPoint))


if __name__ == '__main__':
    main()