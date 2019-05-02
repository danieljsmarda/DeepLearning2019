from torch import nn
from torch.nn import functional as F

class Incept1(nn.Module):
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=2, nb_hidden=200):
        super(Incept1, self).__init__()
        self.name = "Incept1"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_digits)
        self.fc3 = nn.Linear(512, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, nb_classes_pairs)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = F.relu(self.fc1(x0.view(-1, 256)))
        x0 = self.fc2(x0)
        
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = F.relu(self.fc1(x1.view(-1, 256)))
        x1 = self.fc2(x1)
        
        x = torch.cat((x0,x1),1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        
        return x0, x1, x