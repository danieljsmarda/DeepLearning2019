import torch
from torch import *
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F





class AuxModel(nn.Module):

    def __init__(self, nb_hidden=100):
        super(AuxModel, self).__init__()
        self.cl1 = nn.Conv2d(1, 64, kernel_size=3)
        self.cl2 = nn.Conv2d(64, 128, kernel_size=3)
        self.full1 = nn.Linear(512, nb_hidden)
        self.full2 = nn.Linear(nb_hidden,40)
        self.full3 = nn.Linear(40,10)
        self.full4 = nn.Linear(20, 2)
        #self.full5 = nn.Linear(2,1)
        self.criterion = nn
 
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
        #output = torch.max(output, 1)[1]

        return output , channel1, channel2


def train_model(model, optimizer,  train_input, train_target, train_class):

    nb_epochs = 25
    mini_batch_size = 50
    

    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output, c1, c2 = model(train_input.narrow(0, b, mini_batch_size))
            
            target = train_target.narrow(0, b, mini_batch_size)
            target1 = train_class.narrow(0,b,mini_batch_size).narrow(1,0,1)
            target2 = train_class.narrow(0,b,mini_batch_size).narrow(1,1,1)
            #print(c1, target1.view(mini_batch_size), target1.view(mini_batch_size).shape)
            cross = CrossEntropyLoss()
            loss = (cross(c1, target1.view(mini_batch_size)) + cross(c2, target2.view(mini_batch_size))) + cross(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()


        print("Loss at {:3} : {:3}  ".format(e,loss.data.item()))


def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        a,_,_ = model(data_input.narrow(0, b, mini_batch_size))
        val = torch.max(a,1)[1]
        for k in range(mini_batch_size):
            if data_target.data[b + k] != val[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

    