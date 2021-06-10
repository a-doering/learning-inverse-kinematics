import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.functional import Tensor


class RobotArm2d():
    """2D PRRR robot arm from ardizzone et al. (prismatic, rotational, rotational, rotational"""
    def __init__(self, lengths: list = [0.5, 0.5, 1], sigmas: list = [0.25, 0.5, 0.5, 0.5]):
        self.sigmas = torch.FloatTensor(sigmas)
        self.lengths = torch.FloatTensor(lengths)
        self.rangex = (-0.5, 2.5)  # (-0.35, 2.25)
        self.rangey = (-1.5, 1.5)  # (-1.3, 1.3)
        cmap = cm.tab20c
        self.colors = [[cmap(4*c_index), cmap(4*c_index+1), cmap(4*c_index+2)] for c_index in range(5)][-1]
        self.out_dir = "data"
        self.viz_dir = "visualizations"
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        if not os.path.isdir(self.viz_dir):
            os.makedirs(self.viz_dir, exist_ok=True)

    def sample_priors(self, batch_size: int = 1) -> torch.FloatTensor:
        """Normal distributed values of the joint parameters"""
        return torch.randn(batch_size, 4) * self.sigmas

    def advance_joint(self, current_pos: torch.FloatTensor, length: float, angle: float) -> (torch.FloatTensor, torch.FloatTensor):
        """Calculate position of next joint

        :param current_pos: Current position at joint in 2d, size: (n, 2)
        :param length: Length of the arm from current to next position, float
        :param angle: Angle around current joint, float
        :return current_pos: Current position at joint in 2d, size: (n, 2)
        :return next_pos: Next position at end of arm in 2d, size: (n, 2)
        """
        # Cloning is required to not share underlaying data
        next_pos = current_pos.clone()
        #angle = torch.FloatTensor(angle)
        next_pos[:, 0] += length * torch.cos(angle)
        next_pos[:, 1] += length * torch.sin(angle)
        return current_pos, next_pos

    # TODO: look up type annotation if type changes with device
    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Forward kinematics of given joint configurations

        :param thetas: Joint parameters: (n, 4)
        :return: End effector position in 2d: (n, 2)
        """
        p0 = torch.stack([torch.zeros((thetas.shape[0]), device=thetas.device), thetas[:, 0]], axis=1)
        _, p1 = self.advance_joint(p0, self.lengths[0], thetas[:, 1])
        _, p2 = self.advance_joint(p1, self.lengths[1], thetas[:, 1] + thetas[:, 2])
        _, p3 = self.advance_joint(p2, self.lengths[2], thetas[:, 1] + thetas[:, 2] + thetas[:, 3])
        return p3

    def distance_euclidean(self, pos_target: torch.FloatTensor, pos: torch.FloatTensor) -> float:
        """Calculate the pairwise distance between each position and its 
        respective target position, summed up and divided by dimension

        :param pos: Target end effector position, size (n, 2)
        :param thetas: Tensor of end effector positions, size (n, 2)
        :return distance: Mean distance of each position to its respective target position
        """
        pdist = torch.nn.PairwiseDistance(p=2)
        dim = pos.shape[0]
        return torch.sum(pdist(pos_target, pos)) / dim

    def inverse(self, pos: torch.FloatTensor, guesses: torch.FloatTensor, epsilon: float = 5e-2, max_steps: int = 3000, lr: float = 0.2) -> torch.FloatTensor:
        # TODO: implement batch vectorized torch inverse
        raise NotImplementedError

    def init_plot(self) -> plt.figure:
        """Initialize matplotlib figure"""
        return plt.figure(figsize=(8, 8))

    def viz_forward(self, pos: torch.FloatTensor) -> None:
        """Visualization of forward kinematics positions

        :param pos: End effector position in 2d: (n, 2)
        """
        fig = self.init_plot()
        plt.scatter(pos[:, 0], pos[:, 1], s=5)
        plt.xlim(*self.rangex)
        plt.ylim(*self.rangey)
        plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
        plt.title(f"Forward Kinematics with {pos.shape[0]} samples")
        plt.show()

    def viz_inverse(self, pos: torch.FloatTensor, thetas: torch.FloatTensor, save: bool = True, show: bool = False, fig_name: str = "fig", viz_format: tuple = (".png", ".svg")) -> None:
        """Visualization of inverse kinematic configurations for end effector position

        :param pos: End effector position, size (n, 2), use n=1 to get informative plots
        :param thetas: Joint parameters, size (n, 4)
        :param save: Bool, True if plot should be saved
        :param show: Bool, True if plot should be displayed
        :param fig_name: Name of the figure without ending or directory, e.g. "fig1"
        :param viz_format: Formats in which the plot should be saved, e.g. (".png", ".svg") or ("png",)
        """
        # Calculate positions of each joint
        p0 = torch.stack([torch.zeros((thetas.shape[0]), device=thetas.device), thetas[:, 0]], axis=1)
        _, p1 = self.advance_joint(p0, self.lengths[0], thetas[:, 1])
        _, p2 = self.advance_joint(p1, self.lengths[1], thetas[:, 1] + thetas[:, 2])
        _, p3 = self.advance_joint(p2, self.lengths[2], thetas[:, 1] + thetas[:, 2] + thetas[:, 3])

        fig = self.init_plot()

        # Plot arms
        opts = {'alpha': 0.05, 'scale': 1, 'angles': 'xy', 'scale_units': 'xy', 'headlength': 0, 'headaxislength': 0, 'linewidth': 1.0, 'rasterized': True}
        plt.quiver(p0[:, 0], p0[:, 1], (p1-p0)[:, 0], (p1-p0)[:, 1], **{'color': self.colors[0], **opts})
        plt.quiver(p1[:, 0], p1[:, 1], (p2-p1)[:, 0], (p2-p1)[:, 1], **{'color': self.colors[1], **opts})
        plt.quiver(p2[:, 0], p2[:, 1], (p3-p2)[:, 0], (p3-p2)[:, 1], **{'color': self.colors[2], **opts})

        # Plot cross to mark end effector position
        l_cross = 0.6
        plt.vlines(pos[:, 0], pos[:, 1]-l_cross, pos[:, 1]+l_cross, ls='-', colors='gray', linewidth=.5, alpha=.5, zorder=-1)
        plt.hlines(pos[:, 1], pos[:, 0]-l_cross, pos[:, 0]+l_cross, ls='-', colors='gray', linewidth=.5, alpha=.5, zorder=-1)

        plt.xlim(*self.rangex)
        plt.ylim(*self.rangey)
        plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
        # Euclidean position is only calculated to the first entry of pos, while target crosses for all will be displayed
        plt.title(f"Inverse Kinematics with {thetas.shape[0]} samples, mean euc. distance = {self.distance_euclidean(pos, p3):.3f}")

        if save:
            for format in viz_format:
                fig.savefig(os.path.join(self.viz_dir, fig_name) + format)
        if show:
            plt.show()
        plt.close(fig)

    def generate_data(self, thetas: torch.FloatTensor, num_inverses: int) -> None:
        """Generate training data: for each prior the end effector position is
        Calculated and for each end effector position num_inverses will be sampled

        :param thetas: Joint parameters, size (n, 4)
        :param num_inverses: Amount of inverses to be created per end effector position
        """
        pos = self.forward(thetas)
        num_positions = thetas.shape[0]
        guesses = torch.zeros((num_positions * num_inverses, 4))
        # TODO: finish after implementing inverse kinematics
        raise NotImplementedError


if __name__ == "__main__":
    arm = RobotArm2d()
    num_forward = 10
    num_inverse_each = 100
    # Viz forward
    priors = arm.sample_priors(num_forward)
    pos = arm.forward(priors)
    arm.viz_forward(pos)
    # # Viz and test inverse
    # guesses = arm.inverse(pos, arm.sample_priors(num_inverse_each))
    # arm.viz_inverse(pos, guesses)
