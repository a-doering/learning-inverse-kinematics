import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, dataloader
from gan_3d.dataset import InverseDataset3d
from gan_3d.model import Generator, Discriminator, DHead, QHead
from tqdm import tqdm
import wandb
import time
import yaml
from rokin.Robots import JustinArm07
from kinematics.JustinArm07 import JustinArm07Net
from evaluate.plot_3d import viz_robot_line

def set_dataloader(data_path: str, batch_size: str) -> DataLoader:
    return DataLoader(
        InverseDataset3d(
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
        name="3d weight pos = 8, no sigmoid",
        tags=["3d", "pos", "no_rot"],
        config=config
    )
    return wandb.config

def distance_euclidean(pos_target: torch.FloatTensor, pos: torch.FloatTensor) -> float:
        """Calculate the pairwise distance between each position and its 
        respective target position, summed up and divided by dimension

        :param pos: Target end effector position, size (n, 2)
        :param thetas: Tensor of end effector positions, size (n, 2)
        :return distance: Mean distance of each position to its respective target position
        """
        pdist = torch.nn.PairwiseDistance(p=2)
        dim = pos.shape[0]
        return torch.sum(pdist(pos_target, pos)) / dim

def train(config_path: str = "config/config_infogan_3d.yaml") -> None:
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
    discriminator = Discriminator(num_thetas=config.num_thetas, pos_dim=config.pos_dim)
    dhead = DHead()
    qhead = QHead(pos_dim=config.pos_dim, latent_dim=config.latent_dim)
    # Loss for discrimination between real and fake
    adversarial_loss = torch.nn.MSELoss()
    # Loss for continuous latent variables
    continuous_loss = torch.nn.GaussianNLLLoss()
    
    # Print model to log structure
    print(generator)
    print(discriminator)
    print(dhead)
    print(qhead)
    print(device)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        dhead.cuda()
        qhead.cuda()
        adversarial_loss.cuda()
        continuous_loss.cuda()

    # Optimizer
    optimizer_D = torch.optim.Adam([{'params': discriminator.parameters()}, {'params': dhead.parameters()}], lr=config.lr)
    optimizer_G = torch.optim.Adam([{'params': generator.parameters()}, {'params': qhead.parameters()}], lr=config.lr)
    # Log models
    wandb.watch(generator, optimizer_G, log="all", log_freq=10)  # , log_freq=100
    wandb.watch(discriminator, optimizer_G, log="all", log_freq=10)  # , log_freq=100

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Adversarial ground truths
    valid = Tensor(config.batch_size, 1).fill_(1.0)
    fake = Tensor(config.batch_size, 1).fill_(0.0)
    batches_done = 0
    #TODO: make this a parameter of the config
    arm = JustinArm07()
    arm_net = JustinArm07Net()
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(1, config.num_epochs + 1)):
        for iter, (thetas_real, pos_real) in enumerate(dataloader):
            
            # Convert to right device
            thetas_real = thetas_real.to(device)
            pos_real = pos_real.to(device)
            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim)))

            # Train Discriminator and dhead
            # ---------------------
            optimizer_D.zero_grad()

            d_out_real = discriminator(thetas_real, pos_real)
            validity_real = dhead(d_out_real)
            loss_D_real = adversarial_loss(validity_real, valid)

            # Generation of positions can be a random position that can be achieved using forward kinematics of random input
            pos_gen = arm_net.forward(Tensor(arm.sample_q(shape=config.batch_size)))[:, -1, 0:3, 3] # TCP position
            # Generate batch of thetas
            thetas_gen = generator(z, pos_gen)

            # Loss for generated (fake) thetas
            d_out_fake = discriminator(thetas_gen, pos_gen)
            validity_fake = dhead(d_out_fake)
            loss_D_fake = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            # Train Generator and qhead
            # ---------------------
            optimizer_G.zero_grad()

            # Generation of positions can be a random position that can be achieved using forward kinematics of random input
            pos_gen = arm_net.forward(Tensor(arm.sample_q(shape=config.batch_size)))[:, -1, 0:3, 3] # TCP position
            # Generate batch of thetas
            thetas_gen = generator(z, pos_gen)

            # Adversarial loss
            d_out_fake = discriminator(thetas_gen, pos_gen)
            validity = dhead(d_out_fake)
            loss_G_fake = adversarial_loss(validity, valid)

            # Distance based loss
            pos_forward = arm_net.forward(thetas_gen)[:, -1, 0:3, 3] # TCP position
            loss_G_pos = distance_euclidean(pos_gen, pos_forward)

            # Latent loss
            latent_pos, latent_z = qhead(d_out_fake)
            loss_Q_pos = distance_euclidean(pos_gen, latent_pos)

            # TODO: latent loss, add to loss_G and implement... how? no idea
            loss_G = loss_G_fake + config.weight_pos * loss_G_pos + loss_Q_pos
            # Backward step
            loss_G.backward()
            optimizer_G.step()
            batches_done += 1


        # Test the generator, visualize and calculate mean distance
        if epoch % config.sample_interval == 0:
            start = time.time()
            generator.eval()
            # Create test position
            pos_test = torch.full_like(pos_real, fill_value=config.pos_test[0], device=device)
            pos_test[:, 1] = config.pos_test[1]
            pos_test[:, 2] = config.pos_test[2]

            # Create test batch, all with same target position
            z_test = Tensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim)))
            # Inference
            with torch.no_grad():
                generated_test_batch = generator(z_test, pos_test).detach()
            # # Visualize
            fig_name = f"{epoch}"
            # arm.viz_inverse(pos_test.cpu(), generated_test_batch.cpu(), fig_name=fig_name)
            # Calculate distance and log
            pos_forward_test = arm_net.forward(generated_test_batch)[:, :, 0:3, 3] # TCP position
            pos_test_single = torch.tensor([[config.pos_test[0], config.pos_test[1], config.pos_test[2]]], device=device)

            viz_robot_line(pos_forward_test[:20].cpu().numpy(), pos_test_single.cpu().numpy(), viz_dir=config.viz_dir, fig_name=fig_name, epoch=epoch)

            test_distance = distance_euclidean(pos_forward_test[:, -1, :], pos_test)
            wandb.log({
                "plot": wandb.Image(os.path.join(config.viz_dir, fig_name + ".png")),
                "generated_batch": generated_test_batch,
                "test_distance": test_distance
            })
            print(f"Epoch: {epoch}/{config.num_epochs} | Batch: {iter + 1}/{len(dataloader)}")
            print(f"Losses: D: {loss_D.item():.3f} | D fake: {loss_D_fake.item():.3f} |G: {loss_G.item():.3f} | G pos: {loss_G_pos.item():.3f} | G fake: {loss_G_fake.item():.3f}| Q pos: {loss_Q_pos.item():.3f} | Test dis: {test_distance:.3f}")
            print(f"Time for evaluation: {time.time()-start}")
            generator.train()

        # Log every epoch
        wandb.log({
            "Epoch": epoch,
            "loss_G_pos": loss_G_pos,
            "loss_G_fake": loss_G_fake,
            "loss_Q_pos": loss_Q_pos,
            "loss_G": loss_G,
            "loss_D_real": loss_D_real,
            "loss_D_fake": loss_D_fake,
            "loss_D": loss_D
        })
        # Save checkpoint on last epoch and every save_model_interval
        if epoch % config.save_model_interval == 0 or epoch == config.num_epochs:
            checkpoint = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "discriminator": discriminator.state_dict(),
                "dhead": dhead.state_dict(),
                "qhead": qhead.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "loss_G_pos": loss_G_pos,
                "loss_G_fake": loss_G_fake,
                "loss_Q_pos": loss_Q_pos,
                "loss_G": loss_G,
                "loss_D_real": loss_D_real,
                "loss_D_fake": loss_D_fake,
                "loss_D": loss_D                
            }
            log_path = os.path.join(wandb.run.dir, "checkpoints")
            os.makedirs(log_path, exist_ok=True)
            # TODO: investigate difference of saving file in wandb dir with torch vs wandb
            torch.save(checkpoint, os.path.join(log_path,  f"{epoch}_checkpoint.pth"))
            # wandb.save(os.path.join(log_path, f"{epoch}_checkpoint.pth"))
            print(f"{epoch} epoch: saved model")
    print(f"Finished training.")

if __name__ == "__main__":
    train()