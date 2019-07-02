# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from scorecardModel.python37_version import utils_v3
from sklearn.model_selection import train_test_split
import pickle
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
    # for col in columnName:
    #     if col in columnNameRemove:
    #         columnName.remove(col)
    columnName = [col for col in data.columns if col not in columnNameRemove]
    for col in columnName:
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

        # （ii）当取值>5时：用bad rate进行编码，放入连续型变量里
        br_encoding_dict = {}  # 记录按照bad rate进行编码的变量，及编码方式
        for col in more_value_feature:
            br_encoding = utils_v3.BadRateEncoding(df=trainData, col=col, target='target')
            # print(br_encoding)
            trainData[col + '_br_encoding'] = br_encoding['encoding']
            br_encoding_dict[col] = br_encoding['bad_rate']
            more_value_feature.append(col + '_br_encoding')

        # 保存需要用坏样本率编码的变量br_encoding_dict
        br_encoding_dict_file = open(outPath + 'br_encoding_dict.pkl', 'wb')
        pickle.dump(br_encoding_dict, br_encoding_dict_file)
        br_encoding_dict_file.close()

        # （iii）对连续型变量进行分箱，包括（ii）中的变量
        continous_merged_dict = {}
        for col in more_value_feature:
            print('{} is in processing'.format(col))
            # －1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
            if -1 not in set(trainData[col]):
                # 分箱后的最多的箱数
                max_interval = 5
                cutOff = utils_v3.ChiMerge(df=trainData, col=col, target='target', max_interval=max_interval,
                                           special_attribute=[],
                                           minBinPcnt=0)
                print('{}变量的切割点是{}'.format(col, cutOff))
                trainData[col + '_Bin'] = trainData[col].map(
                    lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[]))
                # 检验分箱后的单调性是否满足
                print('正在检验变量{}的单调性'.format(col))
                monotone = utils_v3.BadRateMonotone(trainData, col + '_Bin', 'target')
                while (not monotone):
                    # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                    max_interval -= 1
                    cutOff = utils_v3.ChiMerge(df=trainData, col=col, target='target', max_interval=max_interval,
                                               special_attribute=[], minBinPcnt=0)
                    trainData[col + '_Bin'] = trainData[col].map(
                        lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[]))
                    if max_interval == 2:
                        # 当分箱数为2时，必然单调
                        break
                    monotone = utils_v3.BadRateMonotone(trainData, col + '_Bin', 'target')
                newVar = col + '_Bin'
                trainData[newVar] = trainData[col].map(lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[]))
                var_bin_list.append(newVar)
            else:
                max_interval = 5
                # 如果有－1，则除去－1后，其他取值参与分箱
                cutOff = utils_v3.ChiMerge(trainData, col, 'target', max_interval=max_interval, special_attribute=[-1],
                                           minBinPcnt=0)
                trainData[col + '_Bin'] = trainData[col].map(
                    lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[-1]))
                monotone = utils_v3.BadRateMonotone(trainData, col + '_Bin', 'target', ['Bin -1'])
                while (not monotone):
                    max_interval -= 1
                    # 如果有－1，－1的bad rate不参与单调性检验
                    cutOff = utils_v3.ChiMerge(trainData, col, 'target', max_interval=max_interval,
                                               special_attribute=[-1],
                                               minBinPcnt=0)
                    trainData[col + '_Bin'] = trainData[col].map(
                        lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[-1]))
                    if max_interval == 3:
                        # 当分箱数为3-1=2时，必然单调
                        break
                    monotone = utils_v3.BadRateMonotone(trainData, col + '_Bin', 'target', ['Bin -1'])
                newVar = col + '_Bin'
                trainData[newVar] = trainData[col].map(lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[-1]))
                var_bin_list.append(newVar)
            continous_merged_dict[col] = cutOff
        # 需要保存每个变量的分割点
        continous_merged_dict_file = open(outPath + 'continous_merged_dict.pkl', 'wb')
        pickle.dump(continous_merged_dict, continous_merged_dict_file)
        continous_merged_dict_file.close()


if __name__ == '__main__':
    main()