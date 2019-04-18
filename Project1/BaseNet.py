from torch import nn
from torch.nn import functional as F

input_size = 14*14

class BaseNet1C(nn.Module):
    def __init__(self, nb_classes, nb_hidden=200):
        super(BaseNet1C, self).__init__()
        self.name = "BaseNet1C"
        self.fc1 = nn.Linear(1*input_size, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, nb_classes)
        
    def forward(self, x):
        # every 1x14x14 image is reshaped as a 196 1D tensor, len(x) = batch_size
        x = x.reshape((len(x), -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class BaseNet2C(nn.Module):
    def __init__(self, nb_classes, nb_hidden=200):
        super(BaseNet2C, self).__init__()
        self.name = "BaseNet2C"
        self.fc1 = nn.Linear(2*input_size, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, nb_classes)
        
    def forward(self, x):
        # every 2x14x14 image is reshaped as a 1D tensor
        x = x.reshape((len(x), -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        



