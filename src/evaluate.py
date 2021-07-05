import torch
from gan.model import Generator
from kinematics.robot_arm_2d_torch import RobotArm2d
import os
import numpy as np
import wandb

def load_model(config: wandb.config, checkpoint_name: str, checkpoint_path: str = "wandb/latest-run/files/checkpoints"):
    generator = Generator(num_thetas=config.num_thetas, pos_dim=config.pos_dim, latent_dim=config.latent_dim)
    checkpoint_full_path = os.path.join(checkpoint_path, checkpoint_name)
    checkpoint = torch.load(checkpoint_full_path)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator

# TODO: create latent variable walk
def latent_variable_walk():
    # go through latent variables
    raise NotImplementedError

def inference(generator):
    #perform inference
    raise NotImplementedError

def evaluate(checkpoint_name: str, run_dir: str = "wandb/latest-run"):
    wandb.init(
        project="adlr_gan",
        name="evaluation",
        tags=["evaluation"],
        config=os.path.join(run_dir, "files", "config.yaml")
    )
    config = wandb.config

    # Cuda
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
  
    arm = RobotArm2d(config["robot_arm"]["lengths"], config["robot_arm"]["sigmas"])
    generator = load_model(config, checkpoint_name, os.path.join(run_dir, "files", "checkpoints"))
    if cuda:
        generator.cuda()
    # Create test position
    pos_test = torch.full((config.batch_size, config.pos_dim), fill_value=config.pos_test[0], device=device)
    pos_test[:, 1] = config.pos_test[1]
    # Create test batch, all with same target position
    z_test = Tensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim)))
    # Inference
    with torch.no_grad():
        print(z_test.device, pos_test.device)
        generated_test_batch = generator(z_test, pos_test).detach().cpu()
    # Visualize
    fig_name = f"evaluate"
    arm.viz_inverse(pos_test.cpu(), generated_test_batch.cpu(), fig_name=fig_name)
    # Calculate distance and log
    pos_forward_test = arm.forward(generated_test_batch)
    test_distance = arm.distance_euclidean(pos_forward_test, pos_test.cpu())
    print(test_distance)


if __name__ == "__main__":
    evaluate("250_checkpoint_final.pth", "wandb/run-20210705_123150-1nx0yjlw")



    # arm = RobotArm2d()

    # checkpoint_names = ["151_236000_checkpoint.pth", "130_204000_checkpoint.pth", "97_152000_checkpoint.pth", "66_100000_checkpoint.pth"]
    # for checkpoint_name in checkpoint_names:
    #     generator = load_model(checkpoint_name=checkpoint_name, checkpoint_path="checkpoints_1tazchme/checkpoints")
    #     # Sample latent variable
    #     pos_test = torch.full((config.batch_size, config.dim_pos), fill_value=config.pos_test[0])
    #     pos_test[:, 1] = config.pos_test[1]
    #     for i in range(10):
    #         z_test = torch.FloatTensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim)))
    #         with torch.no_grad():
    #             generated_test_batch = generator(z_test, pos_test)
    #         arm.viz_inverse(pos_test, generated_test_batch, fig_name=f"test_batch_{checkpoint_name}_{i}")
      