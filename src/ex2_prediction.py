import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

        #Decision tree
data_downloads =  pd.read_csv("")
data_uploads = pd.read_csv("")
data_merge = pd.read_csv("")





        #Linear regression

features_d = ['chipsettime', 'mcsindex', 'tbs0', 'tbs1', 'rb0', 'rb1']
target_d = 'throughput'

features_u =['qualitytimestamp', 'mcsindex', 'tbs', 'rbs']
target_u = 'tp_cleaned'

#features_m = 
#target_m = 
