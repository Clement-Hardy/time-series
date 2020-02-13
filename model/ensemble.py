# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:10:46 2020

@author: Hard Med lenovo
"""
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn    
import torch.optim as optim
import sys
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append("..//training")
from metric import accuracy
from neural_network_classification import Dataset_classification, net_classification

class ensemble(nn.Module):
    
    def __init__(self, database, nb_models=20):
        super(ensemble, self).__init__() 
        
        self.GPU = False
        self.database = database
        self.nb_models = nb_models
        
        self.build_model()
        
        
    def build_first_stage(self):
        self.first_stage_models = []
        for i in range(self.nb_models):
            self.first_stage_models.append(net_classification(database=self.database))
        
        self.first_stage_train = [False for i in range(len(self.first_stage_models))]
    
    def build_model(self):
        self.build_first_stage()
        
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(5*self.nb_models,
                                      20)])
        self.layers.extend([nn.Linear(20,
                                      5)])
    
    
    def __build_dataset__(self):
        self.dataset_train = Dataset_classification(database=self.database,
                                                    mode="train")
        self.dataset_validation = Dataset_classification(database=self.database,
                                                         mode="validation")
    
    def __build_dataloader__(self, batch_size):
        self.__build_dataset__()
        
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,
                                                            batch_size=batch_size,
                                                            shuffle=True)
        
        self.dataloader_validation = torch.utils.data.DataLoader(self.dataset_train,
                                                                 batch_size=batch_size,
                                                                 shuffle=True)
        
      
    def pred_last_stage(self, features, features_column_name):
        output = self.first_stage_models[0].predict_data(x=features,
                                             features_column_name=features_column_name)
        for i in range(1,self.nb_models):
            output = torch.cat((output, self.first_stage_models[i].predict_data(x=features,
                                             features_column_name=features_column_name)
                                ),dim=1)
        
        return output
        
    
    def forward(self, x, features_column_name):
        x = self.pred_last_stage(features=x,
                                 features_column_name=features_column_name)
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        x = F.softmax(x, dim=1)
        return x
    
    
    def add_models(self, nb_models):
        for i in range(nb_models):
            self.first_stage_models.append(net_classification(database=self.database))
        
        for i in range(nb_models):
            self.first_stage_train.append(False)
        
        self.nb_models+=nb_models
        
    def fit_first_stage(self, lr=1e-3, iter_per_epoch=40, batch_size=100, optimizer="Adam"):
        bar = tqdm(range(self.nb_models))
        for i in bar:
            bar.set_description("Nb models")
            if not self.first_stage_train[i]:
                self.first_stage_models[i].fit(lr=lr,
                                              iter_per_epoch=iter_per_epoch,
                                              batch_size=batch_size,
                                              optimizer=optimizer,
                                              print_accuracy=False)
                self.first_stage_train[i] = True
        
        
    def __train_epoch__(self, loss, optimizer, iter_per_epoch):
        list_loss = []
        list_targets = None
        list_outputs = None

        for iteration, data in enumerate(self.dataloader_train):
            optimizer.zero_grad()
        
            features, features_column_name, target = data
            features_column_name = np.array(features_column_name)[:,0]
            features = features.float()
            target = target.long()
            if self.GPU:
                features = features.cuda()
                target = target.cuda()
                
            output = self.forward(x=features,
                                  features_column_name=features_column_name)
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
            
            if iteration>iter_per_epoch:
                break;
                
        acc = accuracy(output=list_outputs,
                       target=list_targets,
                       nb_classe=5)
        return np.mean(list_loss), acc
    
    
    def __val_epoch__(self, loss, iter_per_epoch):
        self.eval()
        list_loss = []
        list_targets = None
        list_outputs = None

        for iteration, data in enumerate(self.dataloader_validation):
        
            features, features_column_name, target = data
            features_column_name = np.array(features_column_name)[:,0]
            features = features.float()
            target = target.long()
            if self.GPU:
                features = features.cuda()
                target = target.cuda()
                
            output = self.forward(x=features,
                                  features_column_name=features_column_name)
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
            
            if iteration>iter_per_epoch:
                break;
                
        acc = accuracy(output=list_outputs,
                       target=list_targets,
                       nb_classe=5)
        self.train()
        return np.mean(list_loss), acc
    
    
        
    def fit(self, lr=1e-3, nb_epoch = 20, iter_per_epoch=40, batch_size=100, optimizer="Adam"):
        self.fit_first_stage()
        if self.GPU:
            self.cuda()
        self.__build_dataloader__(batch_size=batch_size)
        
        loss = nn.CrossEntropyLoss()
        if optimizer=="ASGD":
            optimizer = optim.ASGD(self.parameters(),
                                   lr=lr,
                                   t0=200)
        elif optimizer=="Adam":
            optimizer = optim.Adam(self.parameters(),
                                   lr=lr)
        for i in range(nb_epoch):
            train_loss, train_accuracy = self.__train_epoch__(loss=loss,
                                                              optimizer=optimizer,
                                                              iter_per_epoch=iter_per_epoch)
            val_loss, val_accuracy = self.__val_epoch__(loss,
                                                        iter_per_epoch=iter_per_epoch)
            print("Epoch num {}:\n \t train_loss: {},\n \t acc overall: {},\n \t acc class 0: {},\n \t acc class 1: {},\n \t acc class 2: {},\n \t acc class 3: {},\n \t acc class 4: {}, \n \n \t val_loss: {},\n \t acc overall: {},\n \t acc class 0: {},\n \t acc class 1: {},\n \t acc class 2: {},\n \t acc class 3: {},\n \t acc class 4: {}\n\n\n".format(i+1,
                                                                                              np.round(train_loss, 8), np.round(train_accuracy[0], 8),
                                                                                              np.round(train_accuracy[1], 8), np.round(train_accuracy[2], 8),
                                                                                              np.round(train_accuracy[3], 8), np.round(train_accuracy[4], 8), np.round(train_accuracy[5], 8),
                                                                                         np.round(val_loss, 8), np.round(val_accuracy[0], 8),
                                                                                              np.round(val_accuracy[1], 8), np.round(val_accuracy[2], 8),
                                                                                              np.round(val_accuracy[3], 8), np.round(val_accuracy[4], 8), np.round(val_accuracy[5], 8)))
       
        
    