import helpers as help
import random
import math
import torch
from torch import FloatTensor, LongTensor, Tensor

#Setting the seed.
random.seed(7)



''' Generate data '''
inputs, targets = help.generate_disc_data(n=1000)

'''Spliting the Dataset for Training/Validation/Testing.'''

train_data, train_targets, validation_data, validation_targets, test_data, test_targets = help.split_data(inputs, targets, 0.7, 0.1, 0.2)



''' Normalize data '''
mean, std = inputs.mean(), inputs.std()
train_data.sub_(mean).div_(std)
validation_data.sub_(mean).div_(std)
test_data.sub_(mean).div_(std)



''' We initiate the model with this set of parameters.'''
number_of_inputs = 2
number_of_units = 25
number_of_outputs = 2

learning_rate = 0.0001
epochs = 50

model = help.Sequential([help.Linear(number_of_inputs, number_of_units), help.ReLu(), help.Linear(number_of_units, number_of_units), help.ReLu(), help.Linear(number_of_units, number_of_units), help.Tanh(), help.Linear(number_of_units, number_of_outputs), help.Tanh()])


print('Here is the structure of the NN Model : Linear->ReLU->Linear->ReLU->Linear->Tanh->Linear->Tanh.')
print('We will be training the model for {:3} epochs with {:5} for learning rate.'.format(epochs,learning_rate))

''' Training the model'''
model, train_error_list, test_error_list = help.train_model(train_data, train_targets, validation_data, validation_targets, 
                                                    model, learning_rate , epochs)


'''Testing the model'''
help.test_model(model, test_data, test_targets)