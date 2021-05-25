import torch
import torch.nn as nn

#TODO set latent dimensions and dim of the layers
num_thetas = 4
latent_dim = 3


#TODO: add target position
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_features: int, out_features: int, normalize: bool = True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, num_thetas)
            nn.Tanh()
        )

    def forward(self, z):
        thetas = self.model(z)
        return thetas


#TODO: add target position
#TODO: possibly add forward(theta)?
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_thetas, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, thetas):
        validity = self.model(thetas)
        return validity
