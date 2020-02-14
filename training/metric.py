# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:23:31 2019

@author: Hard Med lenovo
"""

import numpy as np 

def accuracy_sklearn(output, label):
    if type(output)!=np.ndarray:
        output = np.array(output)
    if len(output)>1:
        output = output.squeeze()
    if type(label)!=np.ndarray:
        label = np.array(label)
    if len(label)>1:
        label = label.squeeze()
    return {"overall": (np.sum(output==label)/label.shape[0]),
            "class_0": (np.sum((output==label)*(label==0))/np.sum(label==0)),
            "class_1": (np.sum((output==label)*(label==0.25))/np.sum(label==0.25)),
            "class_2": (np.sum((output==label)*(label==0.5))/np.sum(label==0.5)),
            "class_3": (np.sum((output==label)*(label==0.75))/np.sum(label==0.75)),
            "class_4": (np.sum((output==label)*(label==1))/np.sum(label==1))}
    
    
    
    
def accuracy(output, target, nb_classe=4):
    if type(target)!=np.ndarray:
        target = target.detach().cpu().numpy()
    if type(output)!=np.ndarray:
        output = output.detach().cpu().numpy()
        
    if nb_classe>2:
        output = np.argmax(output, 1)
        if len(output.shape)>2:
            output = output.flatten()
            target = target.flatten()
    else:
        output = (output>0.5)*1.
        
    equal = (output==target)
    overall = np.sum(equal)/np.sum(target>-100)
    class_0 = np.sum(equal * (target==0))/np.sum(target==0)
    class_1 = np.sum(equal * (target==1))/np.sum(target==1)
    if nb_classe==2:
        return overall, class_0, class_1
    
    class_2 = np.sum(equal * (target==2))/np.sum(target==2)
    class_3 = np.sum(equal * (target==3))/np.sum(target==3)
    if nb_classe==4:
        return overall, class_0, class_1, class_2, class_3
    
    class_4 = np.sum(equal * (target==4))/np.sum(target==4)
    return overall, class_0, class_1, class_2, class_3, class_4