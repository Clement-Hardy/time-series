# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:10:22 2020

@author: Hard Med lenovo
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class model:
    
    def __init__(self):
        
        self.model = None
        
        
    def fit(self, X_train, y_train):
        
        self.model.fit(X_train, y_train)
        
    
    def predict(self, X_test):
        
        data = self.model.predict(X=X_test)
        return data
        
    
    
class model_random_forest(model):
    
    def __init__(self):
        super(model_random_forest).__init__()
        
        
        self.model = RandomForestClassifier(n_estimators=300,
                                            random_state=0,
                                            verbose=True,
                                            n_jobs=-1)
        
        

class model_gradient_boosting(model):
    
    def __init__(self):
        super(model_gradient_boosting).__init__()
        
        
        self.model = RandomForestClassifier(n_estimators=300,
                                            verbose=True,
                                            n_jobs=-1)