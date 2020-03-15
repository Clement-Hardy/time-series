# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:31:37 2020

@author: Hardy
"""
from data.database import Database
from model.MLP import mlp
from submission.utils import write_submission

if __name__ == '__main__':
    name_model = "mlp"
    
    d = Database()
    X_train, y_train = d.get_training_data()
    X_test, y_test = d.get_validation_data()
    X_live, id_live = d.get_live_data()
    
    if name_model=="mlp":
        model = mlp(weigthed=False,
                    features=True)
        
        acc = model.fit(X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test)
        print(acc)
        
        pred = model.predict(X=X_live)
        predictions = id_live.to_frame()
        predictions["prediction_kazutsugi"] = pred
        write_submission(X=predictions,
                         tournament_number=d.get_tournament_value())
        

