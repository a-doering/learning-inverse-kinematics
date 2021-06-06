import torch
import numpy as np
import random
from kinematics.robot_arm_2d_torch import RobotArm2d
from torch.utils.data import DataLoader
from gan.dataset import InverseDataset2d
from gan.model import Generator, Discriminator
from tqdm import tqdm
import wandb


# Configuration
config = dict(
    seed=123456,
    lr=5e-4,
    latent_dim=3,
    n_discriminator=5,
    num_epochs=30,
    sample_interval=100,
    batch_size=64
)

# Set random seeds
seed = config["seed"]
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup wandb for model tracking
wandb.init(
    project="pytorch-test",
    name="second-test",
    tags=["playground"],
    config=config
)
# Rename for easier access
config = wandb.config

dataloader = DataLoader(
    InverseDataset2d(
        path="data/inverse.pickle"
    ),
    batch_size=config.batch_size,
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
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=config.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=config.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    batches_done = 0
    arm = RobotArm2d()
    for epoch in tqdm(range(config.num_epochs)):
        for iter, (thetas_real, pos_real) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Tensor(config.batch_size, 1).fill_(1.0)
            fake = Tensor(config.batch_size, 1).fill_(0.0)

            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim)))
            # Generation of positions can be a random position that can be achieved using forward kinematics of random input
            pos_gen = arm.forward(arm.sample_priors(config.batch_size))

            # Generate batch of thetas
            thetas_gen = generator(z, pos_real)

            # Calculate loss
            validity = discriminator(thetas_gen, pos_real)
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
            validity_fake = discriminator(thetas_gen.detach(), pos_real)
            loss_D_fake = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            batches_done += 1

            if batches_done % config.sample_interval == 0:
                # Tensor of size (batch_size, 2) with always the same position
                # pos_same = Tensor(batch_size, 2).fill_(1.0)
                # pos_same[:, 0] *= 2
                # pos_same[:, 1] *= 0.5
                # print(pos_real.shape, generator(z, pos_same).detach().shape)
                # print(z.shape)
                # arm.viz_inverse(pos_same, generator(z, pos_same).detach(), fig_name=f"{batches_done}")
                arm.viz_inverse(pos_real, generator(z, pos_real).detach(), fig_name=f"{batches_done}")
                print(f"Epoch: {epoch}/{config.num_epochs} | Batch: {iter + 1}/{len(dataloader)} | D loss: {loss_D.item()} | G loss: {loss_G.item()}")

            wandb.log({
                "Epoch": epoch,
                "loss_D": loss_D,
                "loss_G": loss_G
            })
