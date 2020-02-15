# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:36:42 2020

@author: Hard Med lenovo
"""
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import sys
from tqdm import tqdm
sys.path.append("..//training")
from metric import accuracy_sklearn

class model:
    
    
    def __init__(self, stack=False, stack2=False):
        self.possible_stack_model = ["svc", "LGBM", "Random_Forest",
                                     "Ada", "Bagging", "XGB"]
        self.possible_stack2_model = ["svc", "Ada"]
        self.stack = stack
        self.stack2 = stack2
        self.new_variable = False

        if stack:
            self.__build_stack__()
            self.new_variable = False
        if stack2:
           self.__build_stack2__()
           self.new_variable = False
           self.stack = True
        
        self.variables = {}
            
    def __build_stack__(self):
        self.last_stage = []
        for model in self.possible_stack_model:
            self.last_stage.append(globals()[model]())

    def __build_stack2__(self):
        self.last_stage = []
        for model in self.possible_stack2_model:
            self.last_stage.append(globals()[model](stack=True))
            
            
    def __fit_stack__(self, X, y):
        #bar = tqdm(self.last_stage)
        for model in self.last_stage:
            #bar.set_description("Fit last stage stacking")
            model.fit(X=X,
                      y=y)

        self.fit_last_stage = True
            
            
    def __predict_stack__(self, X):
        X_temp = []
        for model in self.last_stage:
            X_temp.append(np.expand_dims(model.predict_proba(X=X)[:,0],axis=1))
            
        return np.concatenate(X_temp,
                              axis=1)
    
    
    def __generate_variable__(self, X=None, y=None, X_test=None):
        if X is not None:
            X = X.copy()
        if X_test is not None:
            X_test = X_test.copy()
        y=np.array(y)
        class_1 = (y==1)*1
        class_0 = (y==0)*1
        
        if X is not None:
            keys = X.keys()
        
            for key in keys:
                data = np.array(X[key])
                if key+"_diff_class1" not in X.keys():
                    mean_class1 = np.mean(data[class_1])
                    X.insert(0, key+"_diff_class1", data - mean_class1)
                    self.variables[key+"_diff_class1"] = {}
                    self.variables[key+"_diff_class1"]['value'] = mean_class1
                    self.variables[key+"_diff_class1"]['key'] = key
                if key+"_diff_class0" not in X.keys():
                    mean_class0 = np.mean(data[class_0])
                    X.insert(0, key+"_diff_class0", data - mean_class0)
                    self.variables[key+"_diff_class0"] = {}
                    self.variables[key+"_diff_class0"]['key'] = key
                    self.variables[key+"_diff_class0"]['value'] = mean_class0
            
            #X = X.drop(key, axis=1)
        if X_test is not None:
            keys = self.variables.keys()
            for key in keys:
                data = np.array(X_test[self.variables[key]['key']])
                if key not in X_test.keys():
                    X_test.insert(0, key,
                                  data-self.variables[key]['value'])
                #X_test = X_test.drop([self.variables[key]['key']], axis=1)
        
        if X_test is None:
            return X
        elif X is None:
            return X_test
        else:
            return X, X_test
        
        
    def fit(self, X, y, proportion_val=0):    
        if proportion_val>0:
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=proportion_val,
                                                                random_state=1)               
                
        else:
            X_train = X
            y_train = y
            
            
        if self.stack:
            self.__fit_stack__(X=X_train,
                               y=y_train)
            X_train = self.__predict_stack__(X=X_train)
            
        if self.new_variable and not self.stack:
            if proportion_val>0:
                X_train, X_test = self.__generate_variable__(X=X_train,
                                                             y=y_train,
                                                             X_test=X_test)
            else:
                X_train = self.__generate_variable__(X=X_train,
                                                     y=y_train)
                
        self.model.fit(X_train, y_train)
        
        if proportion_val>0:
            y_pred = self.predict(X=X_test)
            return accuracy(output=y_pred,
                            label=y_test)
            
            
        
    def predict_proba(self, X):
        if self.new_variable and not self.stack:
            X = self.__generate_variable__(X_test=X)
        return self.model.predict_proba(X)
    
    
    def predict(self, X):
        if self.stack:
            X = self.__predict_stack__(X=X)
        y = self.predict_proba(X=X)
        y = np.argmax(y, axis=1)
        
        return y
    
    
    def cross_val(self, X, y):
        #X = np.array(X)
        #y = np.array(y)
        
        """
        scores = cross_val_score(estimator=self,
                                 X=X,
                                 y=y,
                                 cv=5,
                                 n_jobs=-1)
        return {"scores": scores, "average": np.mean(scores)}
        """
        scores = {}
        scores['overall'] = []
        scores['class_0'] = []
        scores['class_1'] = []
        kfold = KFold(n_splits=20,
                      random_state=0)
        for train_index, test_index in tqdm(kfold.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.fit(X=X_train,
                     y=y_train,
                     proportion_val=0)

            if self.new_variable:
                X_train, X_test = self.__generate_variable__(X=X_train,
                                                             y=y_train,
                                                             X_test=X_test)
            
            output = self.predict(X=X_test)
            scores['overall'].append(accuracy(output=output,
                                               label=y_test)["overall"])
            scores['class_0'].append(accuracy(output=output,
                                               label=y_test)["class_0"])
            scores['class_1'].append(accuracy(output=output,
                                               label=y_test)["class_1"])
    
        return {"scores overall": scores['overall'],
                "average overall": np.mean(scores['overall']),
                "scores class 0": scores['class_0'],
                "average class 0": np.mean(scores['class_0']),
                "scores class 1": scores['class_1'],
                "average class 1": np.mean(scores['class_1']),
                }
    
    
    def grid_search(self, X, y):
        
        gsearch = GridSearchCV(estimator=self.model,
                               param_grid=self.param_grid,
                               n_jobs=-1,
                               cv=5)
        gsearch.fit(X, y)
        
        return {"best_params": gsearch.best_params_,
                "best_scores": gsearch.best_score_,
                "results": gsearch.cv_results_}
    
    
    def grid_search_param_by_param(self, X, y):
        param = self.param_grid
        best_params = {}
        for key in tqdm(param.keys()):
            self.param_grid = {key: param[key]}
            result = self.grid_search(X=X,
                                      y=y)
            setattr(self.model, key, result["best_params"][key])
            best_params[key] = result["best_params"][key]
        
        self.param_grid = param
        return {"best_params": best_params,
                "best_scores": result["best_scores"]}