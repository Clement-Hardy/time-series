# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:30:17 2020

@author: Hardy
"""

from h2o.automl import H2OAutoML
import h2o
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..//training")
from metric import accuracy, accuracy_sklearn

import numpy as np



class Auto_ML:
    
    def __init__(self):
        h2o.init()
        self.model = H2OAutoML(max_models = 10000,
                               max_runtime_secs=120,
                               max_runtime_secs_per_model=20,
                               seed = 1)
        
        
    def fit(self, X, y, proportion_val=0):
        if proportion_val>0:
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=proportion_val,
                                                                random_state=0)
            dataset_test = X_test.copy()
        else:
            X_train = X
            y_train = y
            
        dataset_train = X_train.copy()
        
        dataset_train.insert(0,"label",y_train)
        dataset_train = h2o.H2OFrame(dataset_train)
        
        if proportion_val>0:
            dataset_test.insert(0,"label",y_test)
            dataset_test = h2o.H2OFrame(dataset_test)
            
        self.model.train(y="label",
                         training_frame=dataset_train)
        
        
        if proportion_val>0:
            y_pred = self.predict(X=dataset_test)
            return accuracy_sklearn(output=y_pred,
                                    label=y_test)
        
            
    def predict(self, X):
        if type(X)==np.array:
            X = h2o.H2OFrame(X)
        y = self.model.predict(test_data=X).as_data_frame()
        #y = (y<1.5)*(y>0.5)*1 + (y<2.5)*(y>1.5)*2 + (y<3.5)*(y>2.5)*3 + (y>3.5)*4
        return y