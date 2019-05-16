
#Imports
import helpers as tool
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import Tensor
from torch import nn
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# Setting the seed.
random.seed(7)

# Generate data
inputs, targets = tool.generate_disc_data(n=1000)

print(inputs.shape, targets.shape)

display_data = False
display_results = False
ReluModel = False
compare = False

# Plot the distribution of data points.
if (display_data == True):
    plt.scatter(inputs[:,0].numpy(), inputs[:,1].numpy(), c = (np.squeeze(targets.numpy())),cmap = 'plasma',alpha=0.8)
    plt.savefig('data_generation.png')
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Distribution of points in our dataset, 1's (in blue) and 0's (in yellow).")
    plt.show()

# Split the dataset for training/validation/testing
train_data, train_targets, validation_data, \
    validation_targets, test_data, test_targets = tool.split_data(
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
EPOCHS = 200

model = tool.Sequential([tool.Linear(NB_INPUTS, NB_UNITS),
                         tool.ReLu(),
                         tool.Linear(NB_UNITS, NB_UNITS),
                         tool.ReLu(),
                         tool.Linear(NB_UNITS, NB_UNITS),
                         tool.Tanh(),
                         tool.Linear(NB_UNITS, NB_OUTPUTS),
                         tool.Tanh()])

if(ReluModel == True):

    model = tool.Sequential([tool.Linear(NB_INPUTS, NB_UNITS),
                             tool.ReLu(),
                             tool.Linear(NB_UNITS, NB_UNITS),
                             tool.ReLu(),
                             tool.Linear(NB_UNITS, NB_UNITS),
                             tool.ReLu(),
                             tool.Linear(NB_UNITS, NB_OUTPUTS),
                             tool.ReLu()])


print('Here is the structure of the NN Model: \
    Linear->ReLU->Linear->ReLU->Linear->Tanh->Linear->Tanh.')
print('We will be training the model for {:3} epochs with {:5} \
     for learning rate.'.format(EPOCHS,LEARNING_RATE))


# Train the model.
model, train_error_list, test_error_list = tool.train_model(train_data,
    train_targets, validation_data, validation_targets, model, LEARNING_RATE,
    EPOCHS)


#Plot the graph with the training and testing error.
if(display_results == True):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Evolution of the training and validation error w.r.t epochs.')
    plt.plot(train_error_list,color='red')
    plt.plot(test_error_list,color='blue')
    plt.legend(['Training Error', 'Validation Error'], loc='best')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(' % of Error')
    plt.savefig('display of results.png')
    plt.show()

# Test the model.
tool.test_model(model, test_data, test_targets)

if(compare == True):

    # We will try to compare the performance of our model with the sklear LR classifiers with default parameters.
    logisticRegr = LogisticRegression(solver='lbfgs',max_iter=300)

    #Training the model.
    logisticRegr.fit(train_data.numpy(),train_targets.numpy().reshape(-1,1))

    #Doing the predictions with trained model.
    predictions = logisticRegr.predict(test_data.numpy())
    print("Shape of prediction",predictions.shape)

    # Use score method to get accuracy of model.
    score = logisticRegr.score(test_data.numpy(), test_targets.numpy())
    print("Here is the accuracy of the Logistic Regression:",100*score,"%")


    # We will try to compare the performance of our model with the sklear RF classifiers with default parameters.

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_data.numpy(),train_targets.numpy().reshape(-1,1))

    #Doing the predictions with trained model.
    predictions = rf.predict(test_data.numpy())
    print("Shape of prediction",predictions.shape)

    # Use score method to get accuracy of model.
    score = rf.score(test_data.numpy(), test_targets.numpy())
    print("Here is the accuracy of the Random RandomForestClassifier:",100*score,"%")

    





