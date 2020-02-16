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
       
    def __init__(self, weigthed=False, sample=True, features=True):
        if weigthed is None:
            self.weigth = np.ones(5)
        else:
            self.weigth = np.random.randint(700,1000,5)/1000.
        
        self.sample = sample
        self.features = features
        self.weigthed = weigthed
        
        
        
    def oversampling_class(self, X, y, class_id, nb_add):
        if type(X)!=np.array:
            X = np.array(X)
        if type(y)!=np.array:
            y = np.array(y)
            if len(y.shape)>1:
                y = y[:,0]
        index_class = (y==class_id)
        data_class = X[index_class,:] 
        for i in range(int(nb_add)):
            index = np.random.randint(0, data_class.shape[0])
            index_insert = np.random.randint(0, X.shape[0])
            X = np.insert(X, index_insert, data_class[index,:], axis=0)
            y = np.insert(y, index_insert, class_id)
        
        return X,y
    
    
    
    def balanced_dataset(self, X, y):
        if type(X)!=np.array:
            X = np.array(X)
        if type(y)!=np.array:
            y = np.array(y)  
            
        sample_per_class = {}
        for i in range(5):
            sample_per_class[i] = np.sum(y==i)
        class_id_min = np.argmin(self.weigth)
        sample_per_class_after = {}
        for i in range(5):
            sample_per_class_after[i] = sample_per_class[class_id_min] * (self.weigth[i]/self.weigth[class_id_min])
            
        for i in range(5):
            if sample_per_class_after[i]-sample_per_class[i]>0:
                X,y = self.oversampling_class(X=X,
                                              y=y,
                                              class_id=i,
                                              nb_add=sample_per_class_after[i]-sample_per_class[i])

        return X,y
        
   
    def __set_features__(self, X):
        self.index_features = np.random.choice(np.arange(0, X.shape[1]), replace=False, size=8)
        self.index_samples = np.random.choice(np.arange(0, X.shape[0]), replace=False, size=3000)
        
        
    def __get_features__(self, X):
        if not hasattr(self, 'index_features'):
            self.__set_features__(X=X)
        if type(X)!=np.array:
            X = np.array(X)            
            
        return X[:,self.index_features]
        
    
    def __get_samples__(self, X, y):
        if type(X)!=np.array:
            X = np.array(X)
        if type(y)!=np.array:
            y = np.array(y)
    
        return X[self.index_samples,:], y[self.index_samples] 
    
    
    def fit(self, X_train, y_train, X_test=None, y_test=None):    
        y_train = y_train*4
        if self.features:
            X_train = self.__get_features__(X=X_train)
        if self.sample:
            X_train, y_train = self.__get_samples__(X=X_train,
                                                    y=y_train)
        if self.weigthed:
            X_train, y_train = self.balanced_dataset(X=X_train,
                                                     y=y_train)
        
        self.model.fit(X_train, y_train*4)
        
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X=X_test)/4
            return accuracy_sklearn(output=y_pred,
                                    label=y_test)
            
            
        
    def predict_proba(self, X):
        if self.features:
            X = self.__get_features__(X=X)
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

    def __init__(self, weigthed=True, sample=True, features=True):
        super(mlp, self).__init__(weigthed=weigthed,
                                  sample=sample,
                                  features=features)
        
        self.model = MLPClassifier(hidden_layer_sizes=10,
                                   max_iter=100000,
                                   alpha=0)
        

class ensemble_mlp(model):
    
    def __init__(self, nb_model=3):
        
        self.models = []
        for i in range(nb_model):
            self.models.append(mlp())
         
        self.model = mlp(weigthed=False,
                         sample=False,
                         features=False)
        
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        for model in tqdm(self.models):
            model.fit(X_train=X_train,
                      y_train=y_train,
                      X_test=X_test,
                      y_test=y_test)
            
        results_train = np.expand_dims(self.models[0].predict(X=X_train), axis=1)
        results_test = np.expand_dims(self.models[0].predict(X=X_test), axis=1)
        for i in range(1,len(self.models)):
            results_train = np.concatenate((results_train,
                                            np.expand_dims(self.models[i].predict(X=X_train),axis=1)), axis=1)
            results_test = np.concatenate((results_test,
                                            np.expand_dims(self.models[i].predict(X=X_test),axis=1)), axis=1)
        
        
        self.model.fit(X_train=results_train,
                       y_train=y_train,
                       X_test=results_test,
                       y_test=y_test)
        
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X=X_test)/4
            return accuracy_sklearn(output=y_pred,
                                    label=y_test)
            
            
            
    def predict_proba(self, X):
        results = np.expand_dims(self.models[0].predict(X=X), axis=1)
        
        for i in range(1,len(self.models)):
            results = np.concatenate((results,
                                      np.expand_dims(self.models[i].predict(X=X),axis=1)), axis=1)
        
        return self.model.predict_proba(results)

    
            
        
      
        
        
            