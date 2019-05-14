import helpers as hs
import random

# Setting the seed.
random.seed(7)

# Generate data
inputs, targets = hs.generate_disc_data(n=1000)

# Split the dataset for training/validation/testing
train_data, train_targets, validation_data, \
    validation_targets, test_data, test_targets = hs.split_data(
        inputs, targets, 0.7, 0.1, 0.2)


# Normalize the data
mean, std = inputs.mean(), inputs.std()
train_data.sub_(mean).div_(std)
validation_data.sub_(mean).div_(std)
test_data.sub_(mean).div_(std)



# We initiate the model with this set of parameters.
NB_INPUTS = 2
NB_UNITS = 25
NB_OUTPUTS = 2
LEARNING_RATE = 0.0001
EPOCHS = 300

model = hs.Sequential([hs.Linear(NB_INPUTS, NB_UNITS),
                         hs.ReLu(),
                         hs.Linear(NB_UNITS, NB_UNITS),
                         hs.ReLu(),
                         hs.Linear(NB_UNITS, NB_UNITS),
                         hs.Tanh(),
                         hs.Linear(NB_UNITS, NB_OUTPUTS),
                         hs.Tanh()])


print('Here is the structure of the NN Model: \
    Linear->ReLU->Linear->ReLU->Linear->Tanh->Linear->Tanh.')
print('We will be training the model for {:3} epochs with {:5} \
     for learning rate.'.format(EPOCHS,LEARNING_RATE))

# Train the model.
model, train_error_list, test_error_list = hs.train_model(train_data,
    train_targets, validation_data, validation_targets, model, LEARNING_RATE,
    EPOCHS)


# Test the model.
hs.test_model(model, test_data, test_targets)