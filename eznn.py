import torch
from torch import nn
import numpy as np


from ez import input_seq, target_seq
print(input_seq)
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


