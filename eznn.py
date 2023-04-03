import torch
from torch import nn
import numpy as np


from ez import *

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
#SUL TUO PORTATILE NON CE L'HAI, QUESTA ROBA E' DAVVERO IMPORTANTE SE LA COMPILI SU COLAB
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        #the following line is for inheritance
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        ###single layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        self.fc = nn.Linear(hidden_dim, output_size)

    def init_hidden(self, batch_size):
        #to me this look a bit simplistic
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden
    
