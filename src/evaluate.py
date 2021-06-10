from gan.model import Generator
import kinematics.robot_arm_2d_torch

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

def evaluate():
    pass

def load_model(checkpoint_name: str, checkpoint_path: str = "wandb/latest-run/files/checkpoints"):
    generator = Generator(num_thetas=config.num_thetas, dim_pos=config.dim_pos, latent_dim=config.latent_dim))
    checkpoint = torch.load(checkpoint_full_path)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator




if __name__ == "__main__"
    