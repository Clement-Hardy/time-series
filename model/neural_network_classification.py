# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:57:01 2019

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


class Dataset_classification(Dataset):
    
    
    def __init__(self, database, mode, features="all"):
        
        self.database = database
        self.mode = mode
        self.features = features
        if self.mode=="train":
            self.nb_row = self.database.get_nb_row_training_data()
        elif self.mode=="validation":
            self.nb_row = self.database.get_nb_row_validation_data()
            
    def __len__(self):
        return self.nb_row
    
    def __getitem__(self, index):
        data, y = self.database.get_training_row(index=index,
                                                 features=self.features)
        
        y = int(y*4)
        #print(np.array(data.keys(), dtype=str))
        return np.array(data, dtype=float), list(data.keys()), y
               


class net_classification(nn.Module):
    
    def __init__(self, database):
        super(net_classification, self).__init__()  
        
        self.database = database
        self.GPU = False
        self.batch_normalization = True
        self.dropout = False
        self.increase_channel = True
        self.class_predict = [0,1,2,3,4]
        #self.nb_features = 310
        self.nb_features = np.random.randint(10,310)
        self.features_names = np.random.choice(self.database.get_features_names(), self.nb_features, replace=False)
        self.nb_layer = np.random.randint(low=9,
                                          high=10)
        
        #self.possible_activation = ["Relu", "Sigmoid", "Tanh"]
        self.possible_activation = ["Relu"]
        
        self.build_model()
        
        
    def __add_layer__(self, input_channel, output_channel):
        self.layers.extend([nn.Linear(input_channel,
                                          output_channel)])
        activation = np.random.choice(self.possible_activation, 1).item()
        if activation=="Relu":
            self.layers.extend([nn.ReLU()])
        elif activation=="Sigmoid":
            self.layers.extend([nn.Sigmoid()])
        elif activation=="Tanh":
            self.layers.extend([nn.Tanh()])
                
        if self.dropout:
            proba = np.random.uniform(low=0.3,
                                       high=0.8)
            self.layers.extend([nn.Dropout(p=proba, inplace=False)])
        if self.batch_normalization:
            self.layers.extend([nn.BatchNorm1d(output_channel)])
        
        
        
    def build_model(self):
            
        self.layers = nn.ModuleList()
        output_channel = np.random.randint(low=10,
                                           high=30)
        input_channel = self.nb_features
        
        for i in range(self.nb_layer-1):
            self.__add_layer__(input_channel=input_channel,
                               output_channel=output_channel)
                
            input_channel = output_channel
            if self.increase_channel:
                output_channel += np.random.randint(low=10,
                                                    high=20)
            else:
                output_channel = np.random.randint(low=20,
                                                   high=100)
        output_channel = len(self.class_predict)
        self.layers.extend([nn.Linear(input_channel,
                                      output_channel)])
            
    def forward(self, x, name_features=None):
      
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        x = F.softmax(x, dim=1)
        return x
    
    
    def __build_dataset__(self):
        self.dataset_train = Dataset_classification(database=self.database,
                                                    mode="train",
                                                    features=self.features_names)
        self.dataset_validation = Dataset_classification(database=self.database,
                                                         mode="validation",
                                                         features=self.features_names)
    
    def __build_dataloader__(self, batch_size):
        self.__build_dataset__()
        
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,
                                                            batch_size=batch_size,
                                                            shuffle=True)
        
        self.dataloader_validation = torch.utils.data.DataLoader(self.dataset_train,
                                                                 batch_size=20000,
                                                                 shuffle=True)
        
        
    def __train_epoch__(self, loss, optimizer, iter_per_epoch):
        list_loss = []
        list_targets = None
        list_outputs = None

        for iteration, data in enumerate(self.dataloader_train):
            optimizer.zero_grad()
        
            features, name_features, target = data
            features = features.float()
            target = target.long()
            if self.GPU:
                features = features.cuda()
                target = target.cuda()
                
            output = self.forward(features,
                                  name_features)
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
        self.GPU = True
        if self.GPU:
            self.cuda()
        list_loss = []
        list_targets = None
        list_outputs = None

        for iteration, data in enumerate(self.dataloader_validation):
            features, name_features, target = data
            features = features.float()
            target = target.long()
            if self.GPU:
                features = features.cuda()
                target = target.cuda()
                
            output = self.forward(features,
                                  name_features)
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
            
            if iteration>-1:
                break;
                
        acc = accuracy(output=list_outputs,
                       target=list_targets,
                       nb_classe=5)
        self.train()
        self.cpu()
        self.GPU = False
        return np.mean(list_loss), acc
    
    
    def fit(self, lr=1e-3, iter_per_epoch=40, nb_epoch = 20, batch_size=1000, optimizer="Adam", print_accuracy=True):
        if self.GPU:
            self.cuda()
        self.__build_dataloader__(batch_size=batch_size)
        
        loss = nn.CrossEntropyLoss()
        #loss = nn.MultiMarginLoss()
        if optimizer=="ASGD":
            optimizer = optim.ASGD(self.parameters(),
                                   lr=lr,
                                   t0=200)
        elif optimizer=="Adam":
            optimizer = optim.Adam(self.parameters(),
                                   lr=lr)
            
        bar = range(nb_epoch)
        if not print_accuracy:
            bar = tqdm(bar)

        for i in bar:
            if not print_accuracy:
                bar.set_description("Epoch")
            train_loss, train_accuracy = self.__train_epoch__(loss=loss,
                                                              optimizer=optimizer,
                                                              iter_per_epoch=iter_per_epoch)
            val_loss, val_accuracy = self.__val_epoch__(loss,
                                                       iter_per_epoch=iter_per_epoch)
            if print_accuracy:
                print("Epoch num {}:\n \t train_loss: {},\n \t acc overall: {},\n \t acc class 0: {},\n \t acc class 1: {},\n \t acc class 2: {},\n \t acc class 3: {},\n \t acc class 4: {}, \n \n \t val_loss: {},\n \t acc overall: {},\n \t acc class 0: {},\n \t acc class 1: {},\n \t acc class 2: {},\n \t acc class 3: {},\n \t acc class 4: {}\n\n\n".format(i+1,
                                                                                                  np.round(train_loss, 8), np.round(train_accuracy[0], 8),
                                                                                              np.round(train_accuracy[1], 8), np.round(train_accuracy[2], 8),
                                                                                              np.round(train_accuracy[3], 8), np.round(train_accuracy[4], 8), np.round(train_accuracy[5], 8),
                                                                                         np.round(val_loss, 8), np.round(val_accuracy[0], 8),
                                                                                              np.round(val_accuracy[1], 8), np.round(val_accuracy[2], 8),
                                                                                              np.round(val_accuracy[3], 8), np.round(val_accuracy[4], 8), np.round(val_accuracy[5], 8)))
            
       
        
    def predict_data(self, x, features_column_name=None):
        if type(x)!=torch.Tensor:
            x = torch.from_numpy(x)
        if len(x.shape)==1:
            x.unsqueeze_(dim=0)
        if features_column_name is not None:
            if type(features_column_name)!=np.ndarray:
                features_column_name = np.array(features_column_name)
            index = [np.argwhere(features_column_name==self.features_names[i]).item() for i in range(len(self.features_names))]
            x = x[:,index]
            
        self.eval()
        x = x.float()
        if self.GPU:
            x = x.cuda()
        result = self.forward(x=x)
        self.train()
        return result