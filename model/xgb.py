# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:32:35 2019

@author: Clement Hardy
"""

import pandas as pd
import numpy as np
import xgboost as xgb


class model_xgb:
    
    def __init__(self):
        
        self.model = xgb.XGBClassifier(max_depth=5,
                                       n_estimators=100,
                                       n_jobs=-1,
                                       silent=False)
        
        
    def fit(self, X_train, y_train):
        
        self.model.fit(X_train, y_train)
        
    
    def predict(self, X_test):
        
        data = self.model.predict(data=X_test)
        return data