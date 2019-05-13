import torch
from torch import nn
from torch.nn import functional as F
from time import *
import numpy as np
from torch.nn import CrossEntropyLoss

class WSModel(nn.Module):

    def __init__(self, nb_hidden=100):
        super(WSModel, self).__init__()
        self.cl1 = nn.Conv2d(1, 64, kernel_size=3)
        self.cl2 = nn.Conv2d(64, 128, kernel_size=3)
        self.full1 = nn.Linear(1024, nb_hidden)
        self.full2 = nn.Linear(nb_hidden,10)
        self.full3 = nn.Linear(10, 2)
 
 
    def forward(self, x):
        a = x[:,0,:,:].view(-1,1,14,14)
        b = x[:,1,:,:].view(-1,1,14,14)

        a = F.relu(F.max_pool2d(self.cl1(a), kernel_size=2, stride=2))
        b = F.relu( F.max_pool2d(self.cl1(b), kernel_size=2, stride=2))
        a = F.relu(F.max_pool2d(self.cl2(a), kernel_size=2, stride=2))
        b = F.relu(F.max_pool2d(self.cl2(b), kernel_size=2, stride=2))
        
 
        output = torch.cat((a.view(-1, 512),b.view(-1, 512)),1)
        output = F.relu(self.full1(output))
        output = F.relu(self.full2(output))
        output = self.full3(output)
        return output

    def test_model(self,model, data_input, data_target, mini_batch_size):

        nb_data_errors = 0
        for b in range(0, data_input.size(0), mini_batch_size):
            a = model(data_input.narrow(0, b, mini_batch_size))
            val = torch.max(a,1)[1]
            for k in range(mini_batch_size):
                if data_target.data[b + k] != val[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors


class WSModel1(nn.Module):

    def __init__(self, nb_hidden=100):
        super(WSModel1, self).__init__()
        self.cl1 = nn.Conv2d(1, 64, kernel_size=3)
        self.cl2 = nn.Conv2d(64, 128, kernel_size=3)
        self.full1 = nn.Linear(1024, nb_hidden)
        self.full2 = nn.Linear(nb_hidden, 2)
 
 
    def forward(self, x):
        a = x[:,0,:,:].view(-1,1,14,14)
        b = x[:,1,:,:].view(-1,1,14,14)

        a = F.relu(F.max_pool2d(self.cl1(a), kernel_size=2, stride=2))
        b = F.relu( F.max_pool2d(self.cl1(b), kernel_size=2, stride=2))
        a = F.relu(F.max_pool2d(self.cl2(a), kernel_size=2, stride=2))
        b = F.relu(F.max_pool2d(self.cl2(b), kernel_size=2, stride=2))
        
 
        output = torch.cat((a.view(-1, 512),b.view(-1, 512)),1)
        output = F.relu(self.full1(output))
        output = self.full2(output)
        return output


    def test_model(self,model, data_input, data_target, mini_batch_size):

        nb_data_errors = 0
        for b in range(0, data_input.size(0), mini_batch_size):
            a = model(data_input.narrow(0, b, mini_batch_size))
            val = torch.max(a,1)[1]
            for k in range(mini_batch_size):
                if data_target.data[b + k] != val[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors
   

def train_model_WS(model, optimizer,  train_input, train_target, epochs,batch_size,type_of_loss):


    nb_epochs = epochs
    mini_batch_size = batch_size
    loss_graph = np.empty([2,nb_epochs])
    acc_epochs = []
    

    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            
            target = train_target.narrow(0, b, mini_batch_size)
            cross = type_of_loss
            loss = cross(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        loss_graph[0][e] = e
        loss_graph[1][e] = loss.data.item()    
        if (e == 0 or e == nb_epochs-1 ):   
            print("Loss at epoch{:3} : {:3}  ".format(e,loss.data.item()))
        acc_epochs.append(model.test_model(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100)
    return loss_graph , acc_epochs  


