# _*_ coding: utf-8 _*_

import pickle
import os

data_path = './data'


trainDataPkl = pickle.load(open(os.path.join(data_path, 'trainData.pkl')))

testDataPkl = pickle.load(open(os.path.join(data_path, 'testData.pkl')))

merge_bin_dict_pkl = pickle.load(open(os.path.join(data_path, 'merge_bin_dict.pkl')))

WOE_dict_pkl = pickle.load(open(os.path.join(data_path, 'WOE_dict.pkl')))