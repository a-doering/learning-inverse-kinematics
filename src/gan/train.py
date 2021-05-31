from torch.utils.data import DataLoader

import numpy as np
from dataset import InverseDataset2d
from model import Generator, Discriminator
from torch.autograd import Variable
import torch

# TODO: select parameters
lr = 5e-4
latent_dim = 3
clip_value = 0.01
n_discriminator = 5
num_epochs = 500
sample_interval = 1000
batch_size = 64

dataloader = DataLoader(
    InverseDataset2d(
        path="data/inverse.pickle"
    ),
    batch_size=batch_size,
    shuffle=True
)


def train():
    generator = Generator()
    discriminator = Discriminator()
    adversarial_loss = torch.nn.MSELoss()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizer
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    batches_done = 0
    for epoch in range(num_epochs):
        for iter, (thetas, targets) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_data = Variable(thetas.type(Tensor))
            targets = Variable(targets.type(Tensor))

            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            # Target generation can be a random position that can be achieved using forward kinematics of random input
            # TODO: replace with real forward from robot_arm_2d_torch, this will currently not work
            # TODO: Merge the kinematics branch into this gan branch
            gen_targets = Variable(Tensor(forward(sample_priors(batch_size))))

            # Generate batch of thetas
            gen_thetas = generator(z, gen_targets)

            # Calculate loss
            validity = discriminator(gen_thetas, gen_targets)
            loss_G = adversarial_loss(validity, valid)

            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real thetas
            validity_real = discriminator(real_data, targets)
            loss_D_real = adversarial_loss(validity_real, valid)

            # Loss for generated (fake) thetas
            validity_fake = discriminator(gen_thetas.detach(), gen_targets)
            loss_D_fake = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            batches_done += 1
            print(f"Epoch: {epoch}/{num_epochs} | Batch: {batches_done % len(dataloader)}/{len(dataloader)} | D loss: {loss_D.item()} | G loss: {loss_G.item()}")
            if batches_done % sample_interval == 0:
                # TODO: add inverse kinematic visualization
                print("Should create visualization here")


if __name__ == "__main__":
    train()
