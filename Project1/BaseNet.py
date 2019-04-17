from torch import nn
from torch.nn import functional as F

class BaseNet(nn.Module):
    def __init__(self, nb_classes, nb_hidden=200):
        super(BaseNet, self).__init__()
        self.name = "BaseNet"
        self.fc1 = nn.Linear(196, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, nb_classes)
        
    def forward(self, x):
        # every 1x14x14 image is reshaped as a 196 1D tensor
        x = x.reshape((len(x), -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



