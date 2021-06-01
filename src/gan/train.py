from  kinematics.robot_arm_2d_torch import RobotArm2d
from torch.utils.data import DataLoader
import numpy as np
from gan.dataset import InverseDataset2d
from gan.model import Generator, Discriminator
import torch


# TODO: select parameters
lr = 5e-4
latent_dim = 3
clip_value = 0.01
n_discriminator = 5
num_epochs = 500
sample_interval = 200
batch_size = 64

dataloader = DataLoader(
    InverseDataset2d(
        path="data/inverse.pickle"
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
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
    arm = RobotArm2d()
    for epoch in range(num_epochs):
        for iter, (thetas_real, pos_real) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Tensor(batch_size, 1).fill_(1.0) #, requires_grad=False
            fake = Tensor(batch_size, 1).fill_(0.0) #, requires_grad=False

            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))
            # Generation of positions can be a random position that can be achieved using forward kinematics of random input
            pos_gen = arm.forward(arm.sample_priors(batch_size))

            # Generate batch of thetas
            thetas_gen = generator(z, pos_gen)

            # Calculate loss
            validity = discriminator(thetas_gen, pos_gen)
            loss_G = adversarial_loss(validity, valid)

            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real thetas
            validity_real = discriminator(thetas_real, pos_real)
            loss_D_real = adversarial_loss(validity_real, valid)

            # Loss for generated (fake) thetas
            validity_fake = discriminator(thetas_gen.detach(), pos_gen)
            loss_D_fake = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            batches_done += 1
            print(f"Epoch: {epoch}/{num_epochs} | Batch: {batches_done % len(dataloader)}/{len(dataloader)} | D loss: {loss_D.item()} | G loss: {loss_G.item()}")
            if batches_done % sample_interval == 0:
                # Tensor of size (batch_size, 2) with always the same position
                pos_same = Tensor(batch_size, 2).fill_(1.0)
                pos_same[:, 0] *= 2
                pos_same[:, 1] *= 0.5
                arm.viz_inverse(pos_same, generator(z, pos_same).detach())
