import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import torch


class RobotArm2d():
    """2D PRRR robot arm from ardizzone et al. (prismatic, rotational, rotational, rotational"""
    def __init__(self, lengths: list = [0.5, 0.5, 1], sigmas: list = [0.25, 0.5, 0.5, 0.5]):
        self.sigmas = torch.FloatTensor(sigmas)
        self.lengths = torch.FloatTensor(lengths)
        self.rangex = (-0.5, 2.5)  # (-0.35, 2.25)
        self.rangey = (-1.5, 1.5)  # (-1.3, 1.3)
        cmap = cm.tab20c
        self.colors = [[cmap(4*c_index), cmap(4*c_index+1), cmap(4*c_index+2)] for c_index in range(5)][-1]
        self.out_dir = "../../data"

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
        next_pos = torch.FloatTensor(current_pos)
        angle = torch.FloatTensor(angle)
        next_pos[:, 0] += length * torch.cos(angle)
        next_pos[:, 1] += length * torch.sin(angle)
        return current_pos, next_pos

    def forward(self, thetas: torch.FloatTensor) -> torch.FloatTensor:
        """Forward kinematics of given joint configurations

        :param thetas: Joint parameters: (n, 4)
        :return: End effector position in 2d: (n, 2)
        """
        p0 = torch.stack([torch.zeros((thetas.shape[0])), thetas[:, 0]], axis=1)
        _, p1 = self.advance_joint(p0, self.lengths[0], thetas[:, 1])
        _, p2 = self.advance_joint(p1, self.lengths[1], thetas[:, 1] + thetas[:, 2])
        _, p3 = self.advance_joint(p2, self.lengths[2], thetas[:, 1] + thetas[:, 2] + thetas[:, 3])
        return p3

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


if __name__ == "__main__":
    arm = RobotArm2d()
    # Viz forward
    priors = arm.sample_priors(1)
    pos = arm.forward(priors)
    arm.viz_forward(pos)
