import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary

from torchtext.vocab import Vectors, GloVe

import nltk
import random

import numpy as np
import matplotlib.pyplot as plt

from utils import train_loop,test_loop

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, \
        vocab_size, embedding_length=2752):
        super(LSTMClassifier,self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.lstm = nn.LSTM(input_size=embedding_length,hidden_size=hidden_size)
        self.label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
    
    def forward(self, input_sentence):
        input = self.word_embeddings(input_sentence)
        input = input.permute(1,0,2)
        if self.batch_size:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size),requires_grad=True)
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size),requires_grad=True)
        else:
            h_0 = Variable(torch.zeros(1, 16, self.hidden_size),requires_grad=True)
            c_0 = Variable(torch.zeros(1, 16, self.hidden_size),requires_grad=True)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])        
        final_output = self.softmax(final_output)
        
        return final_output