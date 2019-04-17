
import torch
from torchvision import datasets

def getModel(model_name):
    return torch.load(model_name)

########################################################
### FRAMEWORK FOR INPUT AS TWO SINGLE CHANNEL IMAGES ###

# In this framework, the network is first trained to recognize the digits of each image from each pair and with the help of the class labels. To do so, we use the class labels provided and use a CrossEntropyLoss to maximize the response of the correct digit. Once the network can predict the digits, we compare the digits and define if they are a pair or not

nb_classes = 10
nb_input_channels = 1



