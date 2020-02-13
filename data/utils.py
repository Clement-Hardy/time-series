# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 00:37:05 2019

@author: Clement Hardy
"""
from pathlib import Path
import pandas as pd
import os
import numpy as np

def load_csv(data_type="train"):
    if data_type not in ["train", "validation", "live"]:
        raise Exception("Expected data_type equal to train or validation or live, not {}".format(data_type))
    if data_type=="train":
        filename = "numerai_training_data.csv"
    elif data_type=="validation" or data_type=="live":
        filename = "numerai_tournament_data.csv"
        
    file = os.path.join(Path('__file__').parent.absolute(),
                        "csv",
                        filename)
    
    data = pd.read_csv(file)
    if data_type=="validation" or data_type=="live":
        data = data.loc[data["data_type"]==data_type]
    return data



def one_hot_encode(data):
    temp = []
    for key in data.keys():
        extend = pd.get_dummies(data[key])
        dict_extend = {}
        for key_extend in extend.keys():
            dict_extend[key_extend] = key
        extend = extend.rename(columns=dict_extend)
        temp.append(extend)
        
    result = pd.concat(temp, axis=1, join='inner')
    return result

def dataframe_to_dict(data):
    result = {}
    for key in data.keys():
        result[key] = np.array(data[key])
    return result