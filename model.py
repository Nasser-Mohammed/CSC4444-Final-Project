import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #layers
        #layer 1: conv layer output: 12 222x222 feature maps
        self.conv1 = nn.Conv2d(3, 12, kernel_size=(5,5), stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        #output: 32 56x56 feature maps
        self.pool1 = nn.MaxPool2d(kernel_size = (5,5), stride = 4)

        #layer 2: conv layer output: 32 54x54
        self.conv2 = nn.Conv2d(12, 24, kernel_size=(5,5), stride=1, padding=1)

        #layer 4: conv layer output: 64 52x52 feature maps
        self.conv3 = nn.Conv2d(24, 36, kernel_size=(5,5), stride = 1, padding = 1)

        #layer 3: pooling layer output: 64 12x12 feature maps
        self.pool2 = nn.MaxPool2d(kernel_size=(5,5), stride = 4)

        
        #self.dropout2 = nn.Dropout(0.4)
        #layer 3: pooling layer output: 32 54x54 feature maps
        #self.pool1 = nn.MaxPool2d(kernel_size=(5,5), stride = 2)

        #layer 5: conv layer output: 64 10x10 feature maps
        self.conv4 = nn.Conv2d(36, 64, kernel_size = (3,3), stride = 1, padding = 0)

        #layer 6: pooling layer output: 64 4x4 feature maps
        self.pool3 = nn.MaxPool2d(kernel_size = (3,3), stride = 2)

        #layer 7: conv layer output: 256 47x47 feature maps
        #self.conv5 = nn.Conv2d(128, 256, kernel_size = (5,5), stride = 1, padding = 0)

        #layer 8: pooling layer output: 256 23x23 feature maps
        #self.pool3 = nn.MaxPool2d(kernel_size = (3,3), stride = 2)

        #layer 9: pooling layer output: 256 11x11 feature maps
        #self.pool4 = nn.MaxPool2d(kernel_size = (3,3), stride = 2)

        #flatten feature maps into linear data
        self.flat = nn.Flatten()

        #layer 10: fully connected layer
        self.fc1 = nn.Linear(64*4*4, 4096)
        self.dropout2 = nn.Dropout(0.3)

        #layer 11: fully connected layer
        self.fc2 = nn.Linear(4096, 1024)
        self.dropout3 = nn.Dropout(.25)
        #layer 12: fully connected layer
        self.fc3 = nn.Linear(1024, 70)



    def forward(self, x):
        #3x224x224 image shape
        #layer 1
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        #layer 2
        x = self.relu(self.conv2(x))

        #layer 3


        #layer 4
        x = self.relu(self.conv3(x))

        x = self.pool2(x)

        #layer 5
        x = self.relu(self.conv4(x))

        #layer 6
        x = self.pool3(x)

        # #layer 7
        # x = nn.ReLU(self.conv5(x))

        # #layer 8
        # x = self.pool3(x)

        # #layer 9
        # x = self.pool4(x)

        #flatten
        x = self.flat(x)

        #layer 10
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)

        #layer 11
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)

        #layer 12
        x = self.fc3(x)
  

        return x








def create_model():
    model = NeuralNet()
    return model


