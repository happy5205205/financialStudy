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
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 计算VIF
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scorecardModel.python37_version import utils_v3

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
        return yr * 12 + mth
    else:
        return 0


def main():
    data_path = './data'
    outPath = './result/'
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    allData = pd.read_csv(os.path.join(data_path, 'application.csv'), encoding='latin1')
    allData['term'] = allData['term'].apply(lambda x: int(x.replace('months', '')))
    # target 处理，loan_status标签中Fully Paid是正常客户  Charged Off是违约客户
    allData['target'] = allData['loan_status'].apply(lambda x: int(x == 'Charged Off'))
    '''
    由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
    '''
    allData['term'] = allData[allData['term'] == 36]

    trainData, testData = train_test_split(allData, test_size=1 / 4)

    # 固话变量
    trainDataFile = open(outPath + 'trainData.pkl', 'wb')
    pickle.dump(trainData, trainDataFile)
    trainDataFile.close()

    testDataFile = open(outPath + 'testData.pkl', 'wb')
    pickle.dump(testData, testDataFile)
    testDataFile.close()

    '''
        第一步数据预处理
        1、数据清洗
        2、格式转换
        3、缺失值处理

    '''

    # 将带%的百分比变为浮点数
    trainData['int_rate_clean'] = trainData['int_rate'].apply(lambda x: float(x.replace('%', '')) / 100)

    # 将工作年限进行转换
    trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)

    # 将desc的缺失作为一种状态，非缺失作为另一种状态
    trainData['desc_clean'] = trainData['desc'].map(DescExisting)

    # 处理日期 earliest_cr_line标签的格式不统一，需要统一格式且转换成python格式
    trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(ConverDataStr)

    trainData['app_date_clean'] = trainData['issue_d'].map(ConverDataStr)

    #  对缺失值进行处理
    trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(MakeupMissing)
    trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(MakeupMissing)
    trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(MakeupMissing)

    '''
        第二步：变量的衍生
    '''
    # 缺少pub_rec_bankruptcies_clean_Bin变量等待排查

    # 考虑申请额度于收入的占比
    trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis=1)
    # 考虑earliest_cr_line到申请日期的跨度，以月份记
    trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean, x.app_date_clean),
                                                      axis=1)

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
    # (i)取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
    merge_bin_dict = {}  # 存放需要合并的变量，以及合并方法
    var_bin_list = []  # 由于某个取值没有好或者坏样本而需要合并的变量

    for col in less_value_feature:
        binBadRate = utils_v3.BinBadRate(df=trainData, col=col, target='target')[0]
        # print('{}的取值根据标签分组不同属性的坏样本比例为{}'.format(col, binBadRate))
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
        num_features.append(col + '_br_encoding')

    # 保存需要用坏样本率编码的变量br_encoding_dict
    br_encoding_dict_file = open(outPath + 'br_encoding_dict.pkl', 'wb')
    pickle.dump(br_encoding_dict, br_encoding_dict_file)
    br_encoding_dict_file.close()

    # （iii）对连续型变量进行分箱，包括（ii）中的变量
    continous_merged_dict = {}
    for col in num_features:
        print('{} is in processing'.format(col))
        # －1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
        if -1 not in set(trainData[col]):
            # 分箱后的最多的箱数
            max_interval = 5
            cutOff = utils_v3.ChiMerge(df=trainData, col=col, target='target', max_interval=max_interval, special_attribute=[],
                                       minBinPcnt=0)
            print('{}变量的切割点是{}'.format(col, cutOff))
            trainData[col+'_Bin'] = trainData[col].map(lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[]))
            # 检验分箱后的单调性是否满足
            print('正在检验变量{}的单调性'.format(col))
            monotone = utils_v3.BadRateMonotone(trainData, col + '_Bin', 'target')
            while (not monotone):
                # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                max_interval -= 1
                cutOff = utils_v3.ChiMerge(df=trainData, col=col, target='target', max_interval=max_interval, special_attribute=[], minBinPcnt=0)
                trainData[col + '_Bin'] = trainData[col].map(lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[]))
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
            cutOff = utils_v3.ChiMerge(trainData, col, 'target', max_interval=max_interval, special_attribute=[-1], minBinPcnt=0)
            trainData[col + '_Bin'] = trainData[col].map(lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[-1]))
            monotone = utils_v3.BadRateMonotone(trainData, col + '_Bin', 'target', ['Bin -1'])
            while (not monotone):
                max_interval -= 1
                # 如果有－1，－1的bad rate不参与单调性检验
                cutOff = utils_v3.ChiMerge(trainData, col, 'target', max_interval=max_interval, special_attribute=[-1],
                                           minBinPcnt=0)
                trainData[col + '_Bin'] = trainData[col].map(lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[-1]))
                if max_interval == 3:
                    # 当分箱数为3-1=2时，必然单调
                    break
                monotone = utils_v3.BadRateMonotone(trainData, col + '_Bin', 'target', ['Bin -1'])
            newVar = col + '_Bin'
            trainData[newVar] = trainData[col].map(lambda x: utils_v3.AssignBin(x, cutOff, special_attribute=[-1]))
            var_bin_list.append(newVar)
        continous_merged_dict[col] = cutOff
    # 需要保存每个变量的分割点
    continous_merged_dict_file = open(outPath+'continous_merged_dict.pkl', 'wb')
    pickle.dump(continous_merged_dict, continous_merged_dict_file)
    continous_merged_dict_file.close()

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
        woe_iv = utils_v3.CalcWOE(trainData, var, 'target')
        WOE_dict[var] = woe_iv['WOE']
        IV_dict[var] = woe_iv['IV']

    # 保存变量每个分箱的WOE值
    WOE_dict_file = open(outPath + 'WOE_dict.pkl', 'wb')
    pickle.dump(WOE_dict, WOE_dict_file)
    WOE_dict_file.close()

    # 保存变量的IV值
    IV_dict_file = open(outPath + 'IV_dict.pkl', 'wb')
    pickle.dump(IV_dict, IV_dict_file)
    IV_dict_file.close()

    # 将变量IV值进行降序排列，方便后续挑选变量
    IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)

    IV_values = [i[1] for i in IV_dict_sorted]
    IV_name = [i[0] for i in IV_dict_sorted]

    plt.bar(x=range(len(IV_name)), height=IV_values, label='feature IV', alpha=0.8, width=0.3)

    # plt.title('feature IV')
    # plt.bar(range(len(IV_values)), IV_values)
    # plt.show()

    '''
    第五步：单变量分析和多变量分析，均基于WOE编码后的值。
    （1）选择IV高于0.01的变量
    （2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
    '''
    high_IV = {k: v for k, v in IV_dict.items() if v >= 0.02}
    high_IV_sorted = sorted(high_IV.items(), key=lambda x: x[1], reverse=True)

    short_list = high_IV.keys()
    short_list_2 = []
    for var in short_list:
        newVar = var+'_WOE'
        trainData[newVar] = trainData[var].map(WOE_dict[var])
        short_list_2.append(newVar)

    # 对于上一步的结果，计算相关系数矩阵，并画出热力图进行数据可视化
    trainDataWOE = trainData[short_list_2]
    # f, ax = plt.figure(figsize=(10, 8))
    corr = trainDataWOE.corr()
    # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)

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
            y1 = high_IV_sorted[j][0]+'_WOE'
            roh = np.corrcoef(trainData[x1], trainData[y1])[0, 1]
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
    multi_analysis_vars_1 = [high_IV_sorted[i][0] + '_WOE' for i in range(cnt_vars) if i not in deleted_index]
    X = np.matrix(trainData[multi_analysis_vars_1])

    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    max_VIF = max(VIF_list)
    print ('最大的VIF是{}'.format(max_VIF))
    multi_analysis = multi_analysis_vars_1

    '''
    第六步：逻辑回归模型。
    要求：
    1，变量显著
    2，符号为负
    '''
    # (1)将多变量分析的后变量带入LR模型中
    y = trainData['target']
    X = trainData[multi_analysis]
    X['intercept'] = [1] * X.shape[0]

    import statsmodels.api as sm
    LR = sm.Logit(y, X).fit()
    print('---'*30)
    summary = LR.summary()
    print(summary)
    print
    pvals = LR.pvalues
    pvals = pvals.to_dict()

    # 有些变量不显著，需要逐步剔除
    varLargeP = {k: v for k, v in pvals.items() if v>=0.1}
    varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    while (len(varLargeP) > 0 and len(multi_analysis) >0):
        # 每次迭代中，剔除最不显著的变量，直到
        # (1) 剩余所有变量均显著
        # (2) 没有特征可选
        varMaxP = varLargeP[0][0]
        print(varMaxP)
        if varMaxP == 'intercept':
            print('the intercept is not significant!')
            break
        y = trainData['target']
        X = trainData[multi_analysis]
        X['intercept'] = [1]*X.shape[0]

        LR = sm.Logit(y, X).fit()
        pvals = LR.pvalues
        pvals = pvals.to_dict()

        # 有些变量不显著，需要逐步剔除
        varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
        varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    print('---' * 30)
    summary = LR.summary()
    print(summary)

    trainData['prob'] = LR.predict(X)

    auc = roc_auc_score(trainData['target'], trainData['prob'])
    print('auc为{}'.format(auc))

    # 将模型保存
    savaModel = open(outPath+'LR_Model_Normal.pkl', 'wb')
    pickle.dump(LR, savaModel)
    savaModel.close()

    ##############################################################################################################
    # 尝试用L1约束
    # use cross validation to select the best regularization parameter
    multi_analysis = multi_analysis_vars_1
    X = trainData[multi_analysis]
    X = np.matrix(X)
    y = trainData['target']
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    model_parameter = {}
    for C_penalty in np.arange(0.005, 0.2, 0.05):
        for bad_weight in  range(2, 101, 2):
            # print({'C_penalty参数为{}和bad_weight参数为{}'.format(C_penalty, bad_weight)})
            LR_model_2 = LogisticRegressionCV(Cs=[C_penalty], penalty='l1', solver='liblinear', class_weight={1: bad_weight, 0: 1})
            LR_model_2_fit = LR_model_2.fit(X_train, y_train)
            y_pred = LR_model_2_fit.predict_proba(X_test)[:, 1]
            scorecard_result = pd.DataFrame({'prob': y_pred, 'target': y_test})
            performance = utils_v3.KS_AR(scorecard_result, score='prob', target='target')
            KS = performance['KS']
            model_parameter[(C_penalty, bad_weight)] = KS

    # 用随机森林法估计变量重要性
    var_WOE_list = multi_analysis_vars_1
    X = trainData[var_WOE_list]
    X = np.matrix(X)
    y = trainData['target']
    y = np.array(y)

    RFC = RandomForestClassifier()
    RFC_Model = RFC.fit(X, y)
    features_rfc = trainData[var_WOE_list].columns
    featureImportance = {features_rfc[i]: RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
    featureImportanceSorted = sorted(featureImportance.items(), key=lambda x: x[1], reverse=True)
    # we selecte the top 10 features
    features_selection = [k[0] for k in featureImportanceSorted[:8]]

    y = trainData['target']
    X = trainData[features_selection]
    X['intercept'] = [1] * X.shape[0]
    LR = sm.Logit(y, X).fit()
    summary = LR.summary()
    print(summary)


if __name__ == '__main__':
    main()