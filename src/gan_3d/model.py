import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_thetas: int = 4, pos_dim: int = 3, latent_dim: int = 3):
        """Initialize Generator

        :param num_thetas: Number of joint parameters
        :param pos_dim: Dimensions of position, e.g. 3 for 3D (x,y,z)
        :param latent_dim: Latent dimension
        """
        super(Generator, self).__init__()

        def block(in_features: int, out_features: int, normalize: bool = True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + pos_dim, 128, normalize=False),
            *block(128, 256),
            #*block(256, 512),
            #*block(512, 1024),
            nn.Linear(256, num_thetas),
            # Tanh would limit the resulting space so that not all radian values can be created
            # nn.Tanh()
        )

    def forward(self, z, pos):
        # Concatenate positions and latent variable
        gen_input = torch.cat((pos, z), -1)
        thetas = self.model(gen_input)
        return thetas


class Discriminator(nn.Module):
    def __init__(self, num_thetas:int = 4, pos_dim: int = 2):
        """Initialize Discriminator

        :param num_thetas: Number of joint parameters
        :param dim_pos: Dimensions of position, e.g. 3 for 3D (x,y,z)
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_thetas + pos_dim, 256),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 256),
            nn.ReLU(inplace=True)
            #nn.Linear(256, 1)
        )

    def forward(self, thetas, pos) -> bool:
        disc_input = torch.cat((thetas, pos), -1)
        out = self.model(disc_input)
        return out


class DHead(nn.Module):
    def __init__(self):
        """Initialize discriminator head to predict real/fake"""
        super(DHead, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 1),
            #nn.Sigmoid()
        )

    def forward(self, out):
        validity = self.model(out)
        return validity


class QHead(nn.Module):
    def __init__(self, pos_dim: int = 3, latent_dim: int = 3):
        """Initialize auxiliary head to predict the latent variables and control variables
        
        :param num_thetas: Number of joint parameters
        :param dim_pos: Dimensions of position, e.g. 3 for 3D (x,y,z)
        """
        super(QHead, self).__init__()

        self.auxiliary = nn.Sequential(
            nn.Linear(256, pos_dim)
        )

        self.latent = nn.Sequential(
            nn.Linear(256, latent_dim)
        )

    def forward(self, out):
        pos = self.auxiliary(out)
        z = self.latent(out)
        return pos, z
