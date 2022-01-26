import torch 
from torch import nn

class Custom(nn.Module):
    
    def __init__(self):
        super(Custom,self).__init__()
        #(feature_map - kernel + 2*padding)/stride + 1
        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels=32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=0.25)
        
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=0.25)
        
        self.conv_3 = nn.Conv2d(in_channels = 64, out_channels=128,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(p=0.25)
        
        self.fc1 = nn.Linear(in_features=128*16*16, out_features = 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        
        
        
        
    def forward(self,x):
        x = self.relu1(self.conv_1(x))
        x = self.bn1(x)
        x = nn.functional.max_pool2d(x,2) # /2
        x = self.dropout1(x)
        
        x = self.relu2(self.conv_2(x))
        x = self.bn2(x)
        x = nn.functional.max_pool2d(x,2) # /2
        x = self.dropout2(x)
        
        x = self.relu3(self.conv_3(x)) 
        x = self.bn3(x)
        x = nn.functional.max_pool2d(x,2) # /2
        x = self.dropout3(x)
        
        x = x.view(-1, 128*16*16)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.bn5(x)
        
        x = self.fc2(x)
        return x