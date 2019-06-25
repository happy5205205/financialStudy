# _*_ coding: utf-8 _*_

import pickle

data_path = './result/'


trainDataPkl = pickle.load(open(data_path+'trainData.pkl','rb'), encoding='latin1')

testDataPkl = pickle.load(open(data_path+'testDataFile.pkl','rb'), encoding='latin1')

merge_bin_dict_pkl = pickle.load(open(data_path+'merge_bin_dict.pkl','rb'), encoding='latin1')

WOE_dict_pkl = pickle.load(open(data_path+'WOE_dict.pkl','rb'), encoding='latin1')

