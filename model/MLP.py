# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:45:35 2020

@author: Hard Med lenovo
"""
from sklearn.neural_network import MLPClassifier



# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:36:42 2020

@author: Hard Med lenovo
"""
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
import sys
from tqdm import tqdm
sys.path.append("..//training")
from metric import accuracy_sklearn



class model:
       
     
    def oversampling_class(self, X, y, class_id, nb_add):
        if type(X)!=np.array:
            X = np.array(X)
        if type(y)!=np.array:
            y = np.array(y)
            
        index_class = (y==class_id)
        data_class = X[index_class,:]
        
        for i in range(nb_add):
            index = np.random.randint(0, data_class.shape[0])
            index_insert = np.random.randint(0, X.shape[0])
            np.insert(X, index_insert, data_class[index,:])
            np.insert(y, index_insert, class_id)
        
        return X,y
    
    
    
    def balanced_dataset(self, X, y):
        if type(X)!=np.array:
            X = np.array(X)
        if type(y)!=np.array:
            y = np.array(y)
            
            
        sample_per_class = {}
        for i in range(4):
            sample_per_class[i] = np.sum(y==i)
        class_id_min = np.argmin(self.weight)
            
        sample_per_class_after = {}
        for i in range(4):
            sample_per_class_after[i] = sample_per_class[class_id_min] * (sample_per_class[i]/sample_per_class[class_id_min])
            
        for i in range(4):
            if sample_per_class_after[i]-sample_per_class[i]>0:
                X,y = self.oversampling_class(X=X,
                                              y=y,
                                              class_id=i,
                                              nb_add=sample_per_class_after[i]-sample_per_class[i])

        return X,y
        
        
            
    def fit(self, X_train, y_train, X_test=None, y_test=None):    
            
                
        self.model.fit(X_train, y_train*4)
        
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X=X_test)/4
            return accuracy_sklearn(output=y_pred,
                                    label=y_test)
            
            
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    
    def predict(self, X):
        y = self.predict_proba(X=X)
        y = np.argmax(y, axis=1)
        
        return y
    
    
    def cross_val(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        scores = cross_val_score(estimator=self,
                                 X=X,
                                 y=y,
                                 cv=5,
                                 n_jobs=-1)
        return {"scores": scores, "average": np.mean(scores)}
        
    
    
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






class mlp(model):

    def __init__(self):
        super(mlp, self).__init__()
        
        self.model = MLPClassifier(hidden_layer_sizes=10,
                                   max_iter=100000,
                                   alpha=0)
        
        