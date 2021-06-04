import torch
import torch.nn as nn

# Number of joint parameters
num_thetas = 4
# Number of targets, 2 for 2D
num_targets = 2
# TODO set latent dimensions and dim of the layers
latent_dim = 3


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # TODO: figure out best embedding for continuous "labels"/targets
        def block(in_features: int, out_features: int, normalize: bool = True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_targets, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, num_thetas),
            # Tanh would limit the resulting space so that not all radian values can be created
            # nn.Tanh()
        )

    def forward(self, z, targets):
        # Concatenate target embeddings and latent variable
        gen_input = torch.cat((targets, z), -1)
        thetas = self.model(gen_input)
        return thetas


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # TODO: figure out embedding for continuous "labels"/targets
        self.model = nn.Sequential(
            nn.Linear(num_thetas + num_targets, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, thetas, targets) -> bool:
        # TODO: check concatination
        disc_input = torch.cat((thetas, targets), -1)
        validity = self.model(disc_input)
        return validity
