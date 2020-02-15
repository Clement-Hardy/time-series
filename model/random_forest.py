# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:10:22 2020

@author: Hard Med lenovo
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_sklearn import model

    
    
class Random_Forest(model):
    
    def __init__(self, stack=False):
        super(Random_Forest, self).__init__(stack=stack)
        
        self.model = RandomForestClassifier(n_estimators=500)
        

