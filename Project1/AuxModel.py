import torch
from torch import *
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import numpy as np





class AuxModel(nn.Module):

    def __init__(self, nb_hidden=100):
        super(AuxModel, self).__init__()

        self.cl1 = nn.Conv2d(1, 64, kernel_size=3)
        self.cl2 = nn.Conv2d(64, 128, kernel_size=3)
        self.full1 = nn.Linear(512, nb_hidden)
        self.full2 = nn.Linear(nb_hidden,40)
        self.full3 = nn.Linear(40,10)
        self.full4 = nn.Linear(20, 2)
 
    def forward(self, x):

        a = x[:,0,:,:].view(-1,1,14,14)
        b = x[:,1,:,:].view(-1,1,14,14)
        a = F.relu(F.max_pool2d(self.cl1(a), kernel_size=2, stride=2))
        b = F.relu(F.max_pool2d(self.cl1(b), kernel_size=2, stride=2))
        a = F.relu(F.max_pool2d(self.cl2(a), kernel_size=2, stride=2))
        b = F.relu(F.max_pool2d(self.cl2(b), kernel_size=2, stride=2))
        a = F.relu(self.full1(a.view(-1, 512)))
        b = F.relu(self.full1(b.view(-1, 512)))
        a = F.relu(self.full2(a))
        b = F.relu(self.full2(b))
        channel1 = F.relu(self.full3(a))
        channel2 = F.relu(self.full3(b))
        
 
        output = torch.cat((channel1,channel2),1)
        output = self.full4(output)

        return output , channel1, channel2


    def test_model(self,model, data_input, data_target, mini_batch_size):

        nb_data_errors = 0
        for b in range(0, data_input.size(0), mini_batch_size):
            a,_,_ = model(data_input.narrow(0, b, mini_batch_size))
            val = torch.max(a,1)[1]
            for k in range(mini_batch_size):
                if data_target.data[b + k] != val[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors





class AuxModel1(nn.Module):

    def __init__(self, nb_hidden=100):
        super(AuxModel1, self).__init__()
        self.cl1 = nn.Conv2d(1, 64, kernel_size=3)
        self.cl2 = nn.Conv2d(64, 128, kernel_size=3)
        self.full1 = nn.Linear(512, nb_hidden)
        self.full2 = nn.Linear(nb_hidden, 10)
        self.full3 = nn.Linear(20,2)
 
 
    def forward(self, x):
        a = x[:,0,:,:].view(-1,1,14,14)
        b = x[:,1,:,:].view(-1,1,14,14)

        a = F.relu(F.max_pool2d(self.cl1(a), kernel_size=2, stride=2))
        b = F.relu( F.max_pool2d(self.cl1(b), kernel_size=2, stride=2))
        a = F.relu(F.max_pool2d(self.cl2(a), kernel_size=2, stride=2))
        b = F.relu(F.max_pool2d(self.cl2(b), kernel_size=2, stride=2))
        a = F.relu(self.full1(a.view(-1, 512)))
        b = F.relu(self.full1(b.view(-1, 512)))

        channel1 = F.relu(self.full2(a))
        channel2 = F.relu(self.full2(b)) 

        output = torch.cat((channel1,channel2),1)
        output = self.full3(output)

        return output , channel1 , channel2

    def test_model(self,model, data_input, data_target, mini_batch_size):

        nb_data_errors = 0
        for b in range(0, data_input.size(0), mini_batch_size):
            a,_,_ = model(data_input.narrow(0, b, mini_batch_size))
            val = torch.max(a,1)[1]
            for k in range(mini_batch_size):
                if data_target.data[b + k] != val[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors


def train_model_AM(model, optimizer, train_input, train_target, train_class,epochs,batch_size,type_of_loss,alpha,beta):

    nb_epochs = epochs
    mini_batch_size = batch_size
    loss_graph = np.empty([2,nb_epochs])
    
    acc_epochs = []
    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output, c1, c2 = model(train_input.narrow(0, b, mini_batch_size))
            
            target = train_target.narrow(0, b, mini_batch_size)
            target1 = train_class.narrow(0,b,mini_batch_size).narrow(1,0,1)
            target2 = train_class.narrow(0,b,mini_batch_size).narrow(1,1,1)
            cross = type_of_loss
            loss = beta*(cross(c1, target1.view(mini_batch_size)) + cross(c2, target2.view(mini_batch_size))) + alpha*cross(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        loss_graph[0][e] = e
        loss_graph[1][e] = loss.data.item()
        acc_epochs.append(model.test_model(model, train_input, train_target, mini_batch_size ) / train_input.size(0) * 100) 
        if (e == 0 or e == nb_epochs-1 ):   
            print("Loss at epoch{:3} : {:3}  ".format(e,loss.data.item()))

    return loss_graph , acc_epochs




    