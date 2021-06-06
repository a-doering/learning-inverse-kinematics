import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_thetas=4, dim_pos=2, latent_dim=3):
        """Initialize Generator

        :param num_thetas: Number of joint parameters
        :param dim_pos: Dimensions of position, e.g. 3 for 3D (x,y,z)
        :param latent_dim: Latent dimension
        """
        super(Generator, self).__init__()

        # TODO: figure out best embedding for continuous "labels"/pos
        def block(in_features: int, out_features: int, normalize: bool = True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + dim_pos, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, num_thetas),
            # Tanh would limit the resulting space so that not all radian values can be created
            # nn.Tanh()
        )

    def forward(self, z, pos):
        # Concatenate positions and latent variable
        gen_input = torch.cat((pos, z), -1)
        thetas = self.model(gen_input)
        return thetas


class Discriminator(nn.Module):
    def __init__(self, num_thetas=4, dim_pos=2):
        """Initialize Discriminator

        :param num_thetas: Number of joint parameters
        :param dim_pos: Dimensions of position, e.g. 3 for 3D (x,y,z)
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_thetas + dim_pos, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, thetas, pos) -> bool:
        disc_input = torch.cat((thetas, pos), -1)
        validity = self.model(disc_input)
        return validity
