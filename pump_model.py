import torch
import torch.nn as nn
import torch.nn.functional as F

class new_Model(nn.Module):
    def __init__(self, in_dim = 10, hidden_dim = 128, n_classes = 8):
        super(new_Model, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_classes)
        
        self.a = torch.rand(8, requires_grad=True)
        self.b = torch.rand(8, requires_grad=True)
        
        
    def forward(self, x, z):
        x = self.a * z[:,0:1] + self.b + x
        #x = torch.cat((x, z[:, 1:]), dim=1)
        x = torch.cat((x, z[:, 1:]), dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        #x = F.sigmoid(x)
      
        return x