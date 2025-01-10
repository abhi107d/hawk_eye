import torch
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description="Program To Convert video into dataset")
    
    # Add required arguments
    parser.add_argument("--x",type=str, required=True, help="input data")
    parser.add_argument("--y",type=int,required=True,help="output data")
    parser.add_argument("--lr",type=float,required=False,default=0.001,help="learning rate")
    # Parse the arguments
    args = parser.parse_args()







# Define the model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.relu(self.fc1(x[:, -1, :]))  # Use the last time step's output
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Trainer():
    def __init__(self,input_size,num_classes,lr):

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)