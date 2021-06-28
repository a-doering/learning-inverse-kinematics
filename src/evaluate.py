import torch
from gan.model import Generator
from kinematics.robot_arm_2d_torch import RobotArm2d
import os
import numpy as np
import wandb

#TODO: load config from wandb
config = dict(
    seed=123456,
    lr=5e-4,
    n_discriminator=5,
    num_epochs=300,
    sample_interval=1000,
    save_model_interval=4000,
    batch_size=64,
    num_thetas=4,
    dim_pos=2,
    latent_dim=3,
    pos_test=[1.51, 0.199]
)
wandb.init(
    project="adlr_gan",
    name="debug_batch",
    tags=["debug_batchwise_generation"],
    config=config
)
config = wandb.config

def evaluate():
    pass

def load_model(checkpoint_name: str, checkpoint_path: str = "wandb/latest-run/files/checkpoints"):
    generator = Generator(num_thetas=config.num_thetas, dim_pos=config.dim_pos, latent_dim=config.latent_dim)
    checkpoint_full_path = os.path.join(checkpoint_path, checkpoint_name)
    checkpoint = torch.load(checkpoint_full_path)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator

# TODO: create latent variable walk
def latent_variable_walk():
    raise NotImplementedError

if __name__ == "__main__":

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    arm = RobotArm2d()

    checkpoint_names = ["151_236000_checkpoint.pth", "130_204000_checkpoint.pth", "97_152000_checkpoint.pth", "66_100000_checkpoint.pth"]
    for checkpoint_name in checkpoint_names:
        generator = load_model(checkpoint_name=checkpoint_name, checkpoint_path="checkpoints_1tazchme/checkpoints")
        # Sample latent variable
        pos_test = torch.full((config.batch_size, config.dim_pos), fill_value=config.pos_test[0])
        pos_test[:, 1] = config.pos_test[1]
        for i in range(10):
            z_test = torch.FloatTensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim)))
            with torch.no_grad():
                generated_test_batch = generator(z_test, pos_test)
            arm.viz_inverse(pos_test, generated_test_batch, fig_name=f"test_batch_{checkpoint_name}_{i}")
      