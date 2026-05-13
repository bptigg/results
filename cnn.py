import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RelaxationCNN(nn.Module):
    def __init__(self):
        super(RelaxationCNN, self).__init__()

class CNN:
    def __init__(self, shape=(3,1000), file = ""):
        if file == "":
            self.shape = shape
            self.build_model()
        else:
            self.model = self.load(file)
        self.name = file

    def build_model(self):
        inputs, size = self.shape
        self.conv1 = nn.Conv1d(in_channels=inputs, out_channels=32,kernel_size=15, padding=7)
        self.attn_gate = nn.Conv1d(32,1,kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32,64,kernel_size=5)
        self.mode_head = nn.Linear(64,3)
        self.time_head = nn.Sequential(nn.Linear(64+3,32),nn.ReLU(), nn.Linear(32,1))

    def forward(self,x):
        x = F.relu(self.conv1(x))
        attn = torch.sigmoid(self.attn_gate(x))
        x = x * attn
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.max(x,dim=2)[0]
        mode_logits = self.mode_head(x)
        mode_prob = F.softmax(mode_logits,dim=1)
        combined = torch.cat((x,mode_prob), dim=1)
        time_out = self.time_head(combined)
        return mode_logits, time_out

    
    def load(self, file):
        return load_model(file + ".keras")
    def save_model(self, file=""):
        weight_dict = self.state_dict()
        if(file==""):
            torch.save(weight_dict, self.name + ".pth")
        else:
            self.name = file
            torch.save(weight_dict, self.name + ".pth")
    
    @staticmethod
    def prepare_input_stack(stack = [],target_len=1000):
        def scale_and_resize(arr):
            resized = np.interp(np.linspace(0,1,target_len),np.linspace(0,1,len(arr)),arr)
            return (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-9)
        
        ch =[]
        if not isinstance(stack[0],(list,np.ndarray)):
            stack = [stack]
        for s in stack:
            ch.append(scale_and_resize(s))
        return np.array(ch,dtype=np.float32)