import torch
from torch import nn
from torch.nn import functional as F
from time import *
import numpy as np
from torch.nn import CrossEntropyLoss


# Base simple network for comparison. 
class SimpleModel(nn.Module):

    def __init__(self, nb_hidden=128):
        super(SimpleModel, self).__init__()
        self.cl1 = nn.Conv2d(2, 64, kernel_size=3)
        self.cl2 = nn.Conv2d(64, 128, kernel_size=3)
        self.full1 = nn.Linear(512, nb_hidden)
        self.full2 = nn.Linear(nb_hidden,2)
 
 
    def forward(self, x):
       
        output = F.relu(F.max_pool2d(self.cl1(x), kernel_size=2, stride=2))
        output = F.relu(F.max_pool2d(self.cl2(output), kernel_size=2, stride=2))
        output = F.relu(self.full1(output.view(-1, 512)))
        output = self.full2(output)

        return output

    # This method is identical in all of the models.
    def test_model(self,model, data_input, data_target, mini_batch_size):

        nb_data_errors = 0
        for b in range(0, data_input.size(0), mini_batch_size):
            a = model(data_input.narrow(0, b, mini_batch_size))
            # val is the output prediction of our model.
            val = torch.max(a,1)[1]

            # Calculate error rate.
            for k in range(mini_batch_size):
                if data_target.data[b + k] != val[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors

# This method is the same in WSModel and WSModel 1, 
# but modified in AuxModel to incorporate the auxiliary loss.
def train_model(model, optimizer, train_input, train_target, epochs, batch_size, type_of_loss):

    nb_epochs = epochs
    mini_batch_size = batch_size

    # Accumulate losses and epoch for plotting later.
    loss_graph = np.empty([2,nb_epochs])
    acc_epochs = []

    for e in range(0,nb_epochs):
        # Split the data into the mini-batches.
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            
            target = train_target.narrow(0, b, mini_batch_size)
            # For most of our trials we used cross-entropy loss but 
            # the type_of_loss parameter allows for different loss functions
            # to be easily used (as specified in test.py).
            cross = type_of_loss
            loss = cross(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Add results to this epoch to the plotting accumulator arrays.
        loss_graph[0][e] = e
        loss_graph[1][e] = loss.data.item()

        # Print results after first and last epoch.
        # To see more or fewer epochs, change the predicate. 
        if (e == 0 or e == nb_epochs-1):
            print("Loss at epoch{:3} : {:3}  ".format(e,loss.data.item()))
        acc_epochs.append(model.test_model(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100)
    # Return the accumulator arrays for plotting.
    return loss_graph, acc_epochs

