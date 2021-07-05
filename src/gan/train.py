import os
import torch
import numpy as np
import random
from kinematics.robot_arm_2d_torch import RobotArm2d
from torch.utils.data import DataLoader, dataloader
from gan.dataset import InverseDataset2d
from gan.model import Generator, Discriminator
from tqdm import tqdm
import wandb
import time
import yaml

# TODO: decide if saving every n epochs or every m samples or batches
def set_dataloader(data_path: str, batch_size: str) -> DataLoader:
    return DataLoader(
        InverseDataset2d(
            path=data_path
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

def set_wandb(config_path: str) -> wandb.config:
    # Configuration
    with open (config_path, "r") as stream:
        config = yaml.safe_load(stream)
    # Setup wandb for model tracking
    wandb.init(
        project="adlr_gan",
        name="debug",
        tags=["device_dataset"],
        config=config
    )
    return wandb.config

def train(config_path: str = "config/config_generator.yaml") -> None:
    config = set_wandb(config_path)
    # Set random seeds
    seed = config["seed"]
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = set_dataloader(config.data_path, config.batch_size)
    generator = Generator(num_thetas=config.num_thetas, pos_dim=config.pos_dim, latent_dim=config.latent_dim)

    # Print model to log structure
    print(generator)
    print(device)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()

    # Optimizer
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=config.lr)

    # Log models
    wandb.watch(generator, optimizer_G, log="all", log_freq=10)  # , log_freq=100

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    batches_done = 0
    arm = RobotArm2d(config["robot_arm"]["lengths"], config["robot_arm"]["sigmas"])
    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(config.num_epochs)):
        for iter, (thetas_real, pos_real) in enumerate(dataloader):
            
            # Convert to right device
            thetas_real = thetas_real.to(device)
            pos_real = pos_real.to(device)

            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim)))
            # Generation of positions can be a random position that can be achieved using forward kinematics of random input
            pos_gen = arm.forward(arm.sample_priors(config.batch_size)).to(device)
            # Generate batch of thetas
            thetas_gen = generator(z, pos_gen)
            # Forward step and calculate loss
            pos_forward = arm.forward(thetas_gen)
            loss_G_pos = arm.distance_euclidean(pos_gen, pos_forward)
            # Backward step
            loss_G_pos.backward()
            optimizer_G.step()
            batches_done += 1

            # Test the generator, visualize and calculate mean distance
            if batches_done % config.sample_interval == 0:
                start = time.time()
                print(f"Epoch: {epoch}/{config.num_epochs} | Batch: {iter + 1}/{len(dataloader)} | G los pos: {loss_G_pos.item()}")#" | Mean Euc: {mean_euclidean}")
                print(f"Time for saving: {time.time()-start}")
            wandb.log({
                "Epoch": epoch,
                "loss_G_pos": loss_G_pos,
            })

    # Save checkpoint of last epoch
    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "loss_G_pos": loss_G_pos,
    }
    log_path = os.path.join(wandb.run.dir, "checkpoints")
    os.makedirs(log_path, exist_ok=True)
    # TODO: investigate difference of saving file in wandb dir with torch vs wandb
    torch.save(checkpoint, os.path.join(log_path,  f"{epoch}_checkpoint_final.pth"))
    # wandb.save(os.path.join(log_path, f"{epoch}_checkpoint.pth"))
    print(f"{epoch} epoch: saved model")
