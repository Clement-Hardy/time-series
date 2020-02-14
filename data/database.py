# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 00:36:47 2019

@author: Clement Hardy
"""
import numpy as np
import pandas as pd
from utils import load_csv, one_hot_encode, dataframe_to_dict, tournament


class Database:
    
    def __init__(self):
        
        self.tournament = tournament()
        self.last_tournament = np.max(self.tournament)
        self.training_data = load_csv(tournament=self.last_tournament,
                                      data_type="train")
        self.validation_data = load_csv(tournament=self.last_tournament,
                                        data_type="validation")
        self.__find_features_name__()
        self.__find_era__()
        self.__find_id__()
        self.one_hot_encode = False
        
    def __find_features_name__(self):
        self.features_names = list(self.training_data.keys()[3:-1])
        
        
    def __find_id__(self):
        self.id_stocks_training = np.unique(self.training_data["id"])
        self.id_stocks_validation = np.unique(self.validation_data["id"])
    
    def __find_era__(self):
        self.era_training = np.unique(self.training_data["era"])
        self.era_validation = np.unique(self.validation_data["era"])
        
    def __dataframe_to_dict__(self):
        self.training_data_dict = dataframe_to_dict(self.training_data)
        self.validation_data_dict = dataframe_to_dict(self.validation_data)
           
    
    def one_hot_encoding(self):
        if not self.one_hot_encode:
            data = self.training_data.drop(["id", "era", "data_type", "target_kazutsugi"], axis=1)
            data = one_hot_encode(data=data)
            self.training_data = pd.concat([self.training_data[["id", "era", "data_type", "target_kazutsugi"]], data], axis=1, join='inner')
            
            data = self.validation_data.drop(["id", "era", "data_type", "target_kazutsugi"], axis=1)
            data = one_hot_encode(data=data)
            self.validation_data = pd.concat([self.validation_data[["id", "era", "data_type", "target_kazutsugi"]], data], axis=1, join='inner')
            
            self.__dataframe_to_dict__()
            self.one_hot_encode = True
            
    
    def get_features_names(self):
        return self.features_names
    
    def get_nb_row_training_data(self):
        return self.training_data.shape[0]
    
    def get_nb_row_validation_data(self):
        return self.validation_data.shape[0]
    
    def __get_data_row__(self, index, data_type, features="all"):
        if not  hasattr(self, "training_data_dict") or not hasattr(self, "validation_data_dict"):           
            if data_type=="train":
                data = self.training_data.iloc[index]
            elif data_type=="validation":
                data = self.validation_data.iloc[index]
            if type(features)==str and features=="all":
                return data[3:-1], data[-1]
            else:
                return data[features], data[-1]
        else:
            data = []
            if features=="all":
                    features = self.features_names
            if data_type=="train":
                for feature in features:
                    data.append(pd.DataFrame(self.training_data_dict[feature][index,:],
                                             index=np.repeat(feature, len(self.training_data_dict[feature][index,:]))).T)
                y = self.training_data_dict["target_kazutsugi"][index]
            elif data_type=="validation":
                for feature in features:
                     #data[feature] = self.validation_data_dict[feature][index,:]
                     data.append(pd.DataFrame(self.validation_data_dict[feature][index,:],
                                             index=np.repeat(feature, len(self.validation_data_dict[feature][index,:]))).T)
                y = self.validation_data_dict["target_kazutsugi"][index]
            return pd.concat(data, axis=1, join='inner').T.squeeze(), y
            
            
    
    def get_training_row(self, index, features="all"):
        return self.__get_data_row__(index=index,
                                     data_type="train",
                                     features=features)
    
    def get_validation_row(self, index, features="all"):
        return self.__get_data_row__(index=index,
                                     data_type="validation",
                                     features=features)
    
    
    def __get_data__(self, data_type, features="all"):
        if data_type=="train":
            data = self.training_data.drop(["id", "era", "data_type", "target_kazutsugi"], axis=1)
            y = self.training_data.loc[:,self.training_data.columns=="target_kazutsugi"]
        elif data_type=="validation":
            data = self.validation_data.drop(["id", "era", "data_type", "target_kazutsugi"], axis=1)
            y = self.validation_data.loc[:,self.validation_data.columns=="target_kazutsugi"]
        if features=="all":
            return data, y
        else:
            return data[:,data.columns.isin(features)], y
        
        
    def get_training_data(self, features="all"):
        return self.__get_data__(features=features,
                                 data_type="train")
    
    def get_validation_data(self, features="all"):
        return self.__get_data__(features=features,
                                 data_type="validation")
    