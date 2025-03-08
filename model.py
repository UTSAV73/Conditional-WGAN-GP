import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

##DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, class_size, embedding_dim):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(class_size, embedding_dim)

        self.seq = nn.Sequential(
            nn.Linear(784 + embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        #adding labels
        label = self.embedding(label)

        #flattening
        input = input.view(input.size(0), -1)

        #concatenate image vector and label
        x = torch.cat([input, label], 1)
        result = self.seq(x)
        return result


## GENERATOR 
class Generator(nn.Module):
    def __init__(self, latent_size, class_size, embedding_dim):
        super(Generator, self).__init__()

        self.embedding = nn.Embedding(class_size, embedding_dim)

        self.seq = nn.Sequential(
            nn.Linear(latent_size + embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

    
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input, label):
        #adding labels
        label = self.embedding(label)
        #concatenate input and labels
        x = torch.cat([input, label], 1)

        result = self.seq(x)
        result = result.view(-1, 1, 28, 28)
        return result