# Import PyTorch tools
from torchvision import datasets
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import dlc_practical_prologue as prologue
from torch.autograd import Variable
from torch.nn import functional as F
# Import external plotting and helper functions
import matplotlib.pyplot as plt
import dlc_practical_prologue as prologue
# Import self-written models and helper functions
from WSharingModel import *
from AuxModel import *
from SimpleCNN import *
from helpers import *



### Import models and load data.
train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(nb=1000)


# normalize it
mean, std = train_input.mean(), train_input.std() 
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

### Treatement of our Training and Testing Data.
train_input, train_target, train_classes = Variable(train_input), Variable(train_target), Variable(train_classes)
test_input, test_target = Variable(test_input), Variable(test_target)


### We define our Learning parameters.
NB_EPOCHS = 50
MINI_BATCH_SIZE = 100
learning_rates = [0.1]


### We define our optimizers and losses and weights for the auxiliary losses.
op = torch.optim.SGD
losses = [CrossEntropyLoss(),BCEWithLogitsLoss()]
alpha = 1
beta = 0.5


### Training of Different Models and Results on Training and Testing set.
train_accs = []
# J index helps determine which training module to use based on the specified model.
for j in range(0,5):
        for i in range(len(learning_rates)):
            models = [WSModel(),WSModel1(),AuxModel(),AuxModel1(),SimpleModel()]
            model = models[j]
            optimizer = op(model.parameters(),lr = learning_rates[i])
            # For the weight-sharing models
            if (j<2):
                loss_aux,acc = train_model_WS(model, optimizer,  train_input, train_target, NB_EPOCHS, MINI_BATCH_SIZE,losses[0])
                train_accs.append(acc)
            # For the Auxiliary Loss models
            elif(j>1 and j!=4):
                loss_aux,acc = train_model_AM(model, optimizer,  train_input, train_target, train_classes,NB_EPOCHS, MINI_BATCH_SIZE,losses[0],alpha,beta)
                train_accs.append(acc)
            # For the Base (Simple) model
            elif(j==4):
                loss_aux,acc = train_model(model, optimizer,  train_input, train_target, NB_EPOCHS, MINI_BATCH_SIZE,losses[0])
                train_accs.append(acc)
                
            print_results(model,op,learning_rates[i],NB_EPOCHS,MINI_BATCH_SIZE, train_input, train_target,test_input, test_target)

            # Uncomment this line if you want to see the evolution the loss duting training.
            #visualize_loss(model,loss_aux,learning_rates[i])

