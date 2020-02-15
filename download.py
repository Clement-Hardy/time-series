# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:37:10 2020

@author: Hardy
"""

import numerapi
import os

if __name__ == '__main__':
    napi = numerapi.NumerAPI(verbosity="info")
    tournament = napi.get_current_round(8)
    if not os.path.isdir(os.path.join("data", "numerai_dataset_{}".format(tournament))):
        napi.download_current_dataset(dest_path="data",
                                      unzip=True)
    if os.path.isfile(os.path.join("data", "numerai_dataset_{}.zip".format(tournament))):
        os.remove(os.path.join("data", "numerai_dataset_{}.zip".format(tournament)))
    
    for file in os.listdir(os.path.join("data", "numerai_dataset_{}".format(tournament))):
        if "numerai" not in file:
            os.remove(os.path.join("data", "numerai_dataset_{}".format(tournament), file))
            
            
    files_names = os.listdir("data")