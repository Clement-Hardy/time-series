# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 01:15:41 2020

@author: Hardy
"""
from pathlib import Path
import pandas as pd
import os



def write_submission(X, tournament_number):
    try:
        path = Path(__file__).parent.absolute()
    except:
        path = Path('__file__').parent.absolute()
        
    file_name = os.path.join(path, "submission_{}.csv".format(tournament_number))
    
    X.to_csv(file_name, index=False)