import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

# Neural Net with convolutions and without weight sharing (non-siamese) nor auxiliary loss
class NetBase(nn.Module):
    def __init__(self):
        super(NetBase, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3_1 = nn.Linear(256, 128)
        self.fc3_2 = nn.Linear(256, 128)
        self.fc4_1 = nn.Linear(128, 10)
        self.fc4_2 = nn.Linear(128, 10)
        self.fc5 = nn.Linear(20, 128)
        self.fc6 = nn.Linear(128, 2)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x[:, 0].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv1_2(x[:, 1].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2_1(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2_2(x2), kernel_size=2, stride=2))
        x1 = F.relu(self.fc3_1(x1.view(-1, 256)))
        x2 = F.relu(self.fc3_2(x2.view(-1, 256)))
        x1 = F.relu(self.fc4_1(x1))
        x2 = F.relu(self.fc4_2(x2))
        x = F.relu(self.fc5(torch.cat((x1, x2), 1)))
        x = self.fc6(x)
        return x, None, None
    
# Neural Net with convolutions and weight sharing (siamese) and without auxiliary loss
class NetWS(nn.Module):
    def __init__(self):
        super(NetWS, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.fc5 = nn.Linear(20, 128)
        self.fc6 = nn.Linear(128, 2)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x[:, 0].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv1(x[:, 1].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x1 = F.relu(self.fc3(x1.view(-1, 256)))
        x2 = F.relu(self.fc3(x2.view(-1, 256)))
        x1 = F.relu(self.fc4(x1))
        x2 = F.relu(self.fc4(x2))
        x = F.relu(self.fc5(torch.cat((x1, x2), 1)))
        x = self.fc6(x)
        return x, None, None
    
# Neural Net with convolutions and auxiliary loss and without weight sharing (non-siamese)
class NetAL(nn.Module):
    def __init__(self):
        super(NetAL, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3_1 = nn.Linear(256, 128)
        self.fc3_2 = nn.Linear(256, 128)
        self.fc4_1 = nn.Linear(128, 10)
        self.fc4_2 = nn.Linear(128, 10)
        self.fc5 = nn.Linear(20, 128)
        self.fc6 = nn.Linear(128, 2)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x[:, 0].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv1_2(x[:, 1].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2_1(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2_2(x2), kernel_size=2, stride=2))
        x1 = F.relu(self.fc3_1(x1.view(-1, 256)))
        x2 = F.relu(self.fc3_2(x2.view(-1, 256)))
        x1_aux = self.fc4_1(x1)
        x2_aux = self.fc4_2(x2)
        x1 = F.relu(x1_aux)
        x2 = F.relu(x2_aux)
        x = F.relu(self.fc5(torch.cat((x1, x2), 1)))
        x = self.fc6(x)
        return x, x1_aux, x2_aux
    
# Neural Net with convolutions and both weight sharing (siamese) and auxiliary loss
class NetWSAL(nn.Module):
    def __init__(self):
        super(NetWSAL, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.fc5 = nn.Linear(20, 128)
        self.fc6 = nn.Linear(128, 2)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x[:, 0].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv1(x[:, 1].view(-1, 1, 14, 14)), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x1 = F.relu(self.fc3(x1.view(-1, 256)))
        x2 = F.relu(self.fc3(x2.view(-1, 256)))
        x1_aux = self.fc4(x1)
        x2_aux = self.fc4(x2)
        x1 = F.relu(x1_aux)
        x2 = F.relu(x2_aux)
        x = F.relu(self.fc5(torch.cat((x1, x2), 1)))
        x = self.fc6(x)
        return x, x1_aux, x2_aux
