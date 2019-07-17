# _*_ coding: utf-8 _*_

import pandas as pd
import numpy as np
import os
import re
import datetime
import time


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

def main():
    data_path = './data'
    out_path = './result'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    allData = pd.read_csv(os.path.join(data_path, 'application.csv'), encoding='latin1')
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


if __name__ == '__main__':
    main()
