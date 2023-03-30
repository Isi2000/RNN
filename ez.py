import torch
from torch import nn

#####import nltk
#####from nltk.book import text1
import numpy as np

#####first he does something similar to what you did in 2 letters

#text = text1

text = ['hey how are you','good i am fine','have a nice day']
chars = set(''.join(text))

int2char = dict(enumerate(chars))
print(int2char)
char2int = {char: ind for ind, char in int2char.items()}

#padding
maxlen = len(max(text, key=len))


for i in range(len(text)):
  while len(text[i])<maxlen:
      text[i] += ' '


# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    input_seq.append(text[i][:-1])
    target_seq.append(text[i][1:])



#The target sequence will always be one-time step ahead of the input sequence.
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]
print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))
    
