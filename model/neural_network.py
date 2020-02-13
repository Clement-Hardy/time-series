# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:29:14 2019

@author: Hard Med lenovo
"""

from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn    
import torch.nn.functional as f  
import torch.optim as optim


class Dataset1(Dataset):
    
    
    def __init__(self, database, mode):
        
        self.database = database
        self.mode = mode
        if self.mode=="train":
            self.nb_row = self.database.get_nb_row_training_data()
        elif self.mode=="validation":
            self.nb_row = self.database.get_nb_row_validation_data()
            
    def __len__(self):
        return self.nb_row
    
    def __getitem__(self, index):
        data, y = self.database.get_training_row(index=index)
        return np.array(data, dtype=float), y
    
    
    

class net_regression(nn.Module):
    
    def __init__(self, database):
        super(net_regression, self).__init__()  
        
        self.database = database
        self.GPU = False
        self.nb_features = 310
        self.nb_layer = np.random.randint(low=2,
                                          high=100)
        self.layers = nn.ModuleList()
        output_channel = np.random.randint(low=20,
                                           high=50)
        input_channel = self.nb_features
        
        for i in range(self.nb_layer):
            self.layers.extend([nn.Linear(input_channel,
                                          output_channel)])
            input_channel = output_channel
            if (i+2)==self.nb_layer:
                output_channel = 1
            else:
                output_channel = np.random.randint(low=5,
                                                   high=15)
            
    def forward(self, x):
      
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
                
        return x
    
    
    def __build_dataset__(self):
        self.dataset_train = Dataset1(database=self.database,
                                      mode="train")
        self.dataset_validation = Dataset1(database=self.database,
                                           mode="validation")
    
    def __build_dataloader__(self):
        self.__build_dataset__()
        
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)
        
        self.dataloader_validation = torch.utils.data.DataLoader(self.dataset_train,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True)
        
        
    def __train_epoch__(self, loss, optimizer):
        list_loss = []
        list_targets = None
        list_outputs = None

        for iteration, data in enumerate(self.dataloader_train):
            optimizer.zero_grad()
        
            features, target = data
            features = features.float()
            if self.GPU:
                features = features.cuda()
                target = target.cuda()
                
            output = self.forward(features)
            output = output.squeeze()
            l = loss(output, target)
        
            list_loss.append(l.item())
            
            l.backward()
            optimizer.step()
        
            if list_targets is None:
                list_targets = target
            else:
                list_targets = torch.cat((list_targets, target), 0)
            if list_outputs is None:
                list_outputs = output
            else:
                list_outputs = torch.cat((list_outputs, output), 0)
        
            del l
            del output
            
            if iteration>200:
                break;

        return np.mean(list_loss)
    
    
    def __val_epoch__(self, loss):
        self.eval()
        list_loss = []
        list_targets = None
        list_outputs = None

        for iteration, data in enumerate(self.dataloader_validation):
        
            features, target = data
            features = features.float()
            if self.GPU:
                features = features.cuda()
                target = target.cuda()
                
            output = self.forward(features)
            output = output.squeeze()
            l = loss(output, target)
        
            list_loss.append(l.item())
            
            l.backward()
        
            if list_targets is None:
                list_targets = target
            else:
                list_targets = torch.cat((list_targets, target), 0)
            if list_outputs is None:
                list_outputs = output
            else:
                list_outputs = torch.cat((list_outputs, output), 0)
        
            del l
            del output
            
            if iteration>200:
                break;
        print(list_outputs)
        self.train()
        return np.mean(list_loss)
    
    
    def fit(self):
        self.batch_size = 30
        self.__build_dataloader__()
        nb_epoch = 1000
        
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(),
                               lr=1e-4)
        for i in range(nb_epoch):
            loss_train = self.__train_epoch__(loss=loss,
                                              optimizer=optimizer)
            loss_validation = self.__val_epoch__(loss)
            print("Train loss: {}\n Validation loss: {}".format(loss_train, loss_validation))
            