import torch
from gan.model import Generator
from kinematics.robot_arm_2d_torch import RobotArm2d
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt

class Evaluator():
    """Evaluation class for the GAN"""
    def __init__(self, checkpoint_name: str, run_dir: str = "wandb/latest-run"):
        self.checkpoint_name = checkpoint_name
        self.run_dir = run_dir        
        wandb.init(
            project="adlr_gan",
            name="evaluation",
            tags=["evaluation"],
            config=os.path.join(self.run_dir, "files", "config.yaml")
        )
        self.config = wandb.config
        self.checkpoint_path = os.path.join(self.run_dir, "files", "checkpoints")
        self.viz_dir = os.path.join("visualizations", os.path.basename(os.path.normpath(run_dir)))
        self.cuda = True if torch.cuda.is_available() else False
        if self.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def load_model(self):
        """"Load the generator with its settings from checkpoint and set to eval mode"""
        self.generator = Generator(num_thetas=self.config.num_thetas, pos_dim=self.config.pos_dim, latent_dim=self.config.latent_dim)
        checkpoint_full_path = os.path.join(self.checkpoint_path, self.checkpoint_name)
        checkpoint = torch.load(checkpoint_full_path)
        self.generator.load_state_dict(checkpoint["generator"])
        self.generator.eval()
        if self.cuda:
            self.generator.cuda()

    def latent_variable_walk(self):
        # TODO: e.g. 3x3 subplots with different latent variables
        raise NotImplementedError

    def plot_multiple_pos(self, n_rows: int, n_cols: int, z: list = None, positions_x = [0.5, 2, 3.5, 5], positions_y = [1.2, 1.2, 1.2, 1.2], save: bool = True, show: bool = False, fig_name: str = "evaluate_multiple_pos", viz_format: tuple = (".png", ".svg")):
        # Different positions, not latent variable walk
        fig, axs = plt.subplots(n_rows,n_cols, figsize=(n_cols * 3, n_rows * 2), facecolor="w", edgecolor="k", sharey=True)
        fig.subplots_adjust(hspace = .5, wspace=.001)        
        axs = axs.ravel()
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        for i in range(n_rows * n_cols):
            ####
            print(i)
            pos_test = torch.full((self.config.batch_size, self.config.pos_dim), fill_value=positions_x[i], device=self.device)
            pos_test[:, 1] = positions_y[i]
            # Create test batch, all with same target position
            z_test = Tensor(np.random.normal(0, 1, (self.config.batch_size, self.config.latent_dim)))
            # Inference
            with torch.no_grad():
                thetas_generated = self.generator(z_test, pos_test).detach().cpu()
            _, distance = self.arm.viz_inverse(pos_test.cpu(), thetas_generated, fig_name=fig_name + f"{i}", ax=axs[i])
            axs[i].set_title(f"d = {distance:.3f}")
        fig.suptitle(f'Average distance to target for arm with {len(self.config["robot_arm"]["sigmas"])} DOF of length {sum(self.config["robot_arm"]["lengths"])}')
        plt.tight_layout()
        if save:
            for format in viz_format:
                fig.savefig(os.path.join(self.viz_dir, fig_name) + format)
                print("save!")
        if show:
            plt.show()
        plt.close(fig)            

    def plot_inverse(self, z, pos):
        raise NotImplementedError

    def calculate_distance(self, thetas, pos):
        pos_forward = self.arm.forward(thetas)
        return self.arm.distance_euclidean(pos_forward, pos.cpu())

    def evaluate(self):
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.arm = RobotArm2d(self.config["robot_arm"]["lengths"], self.config["robot_arm"]["sigmas"], viz_dir = self.viz_dir)
        self.load_model()

        # Create test position
        positions_x = [0.5, 2, 3.5, 5]
        positions_y = [1.2, 1.2, 1.2, 1.2]

        self.plot_multiple_pos(1,4)
        # for i in range(len(positions_x)):
        #     pos_test = torch.full((self.config.batch_size, self.config.pos_dim), fill_value=positions_x[i], device=self.device)
        #     pos_test[:, 1] = positions_y[i]
        #     # Create test batch, all with same target position
        #     z_test = Tensor(np.random.normal(0, 1, (self.config.batch_size, self.config.latent_dim)))
        #     # Inference
        #     with torch.no_grad():
        #         generated_test_batch = self.generator(z_test, pos_test).detach().cpu()
        #     # Visualize
        #     fig_name = f"evaluate_{pos_test[0][0]}_{pos_test[0][1]:.3f}"
        #     #self.arm.viz_inverse(pos_test.cpu(), generated_test_batch.cpu(), fig_name=fig_name)
        #     self.create_subplots(2, 5, pos_test.cpu(), generated_test_batch.cpu(), fig_name=fig_name)
        #     break
        #     # Calculate distance and log
            # print(self.calculate_distance(generated_test_batch, pos_test).item())
 

if __name__ == "__main__":
    evaluator = Evaluator("250_checkpoint_final.pth", "wandb/run-20210705_123150-1nx0yjlw")
    evaluator.evaluate()
