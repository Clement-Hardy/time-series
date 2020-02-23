# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:37:10 2020

@author: Hardy
"""

import numerapi
import os
import shutil
from tqdm import tqdm
"""
class numer(numerapi.NumerAPI):
    
    def __init__(self):
        super(numer, self).__init__()
    
    
    def download_past_dataset(self, dest_path="data",
                                 unzip=True, tournament=8):
        try:
            round_number = self.get_current_round(tournament)
        except ValueError:
            round_number = "x"
            
        url_current = self.get_dataset_url(tournament)
        print(url_current)
        for rounds in tqdm(range(199,200)):
            dest_filename = "numerai_dataset_{0}.zip".format(rounds)

            dataset_path = os.path.join(dest_path, dest_filename)
            if not os.path.isfile(dataset_path):
                numerapi.utils.ensure_directory_exists(dest_path)

                url = url_current.replace(str(round_number), str(rounds))
                numerapi.utils.download_file(url, dataset_path, self.show_progress_bars)
    
"""   
def zip_old_tournament(last_tournament):
    folders = [folder for folder in os.listdir("data") if os.path.isdir(os.path.join("data",folder))]
    for folder in folders:
        if not os.path.isfile(os.path.join("data", folder+".zip")):
            if "numerai_dataset_" in folder and ".zip" not in folder and str(last_tournament) not in folder:
                shutil.make_archive(os.path.join("data", folder), 'zip', os.path.join("data", folder))
    
        if os.path.isdir(os.path.join("data", folder)) and str(last_tournament) not in folder:
            shutil.rmtree(os.path.join("data", folder))
            

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
    
