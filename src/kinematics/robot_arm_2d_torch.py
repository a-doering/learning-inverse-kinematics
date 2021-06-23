import os
import pickle
import matplotlib.pyplot as plt
import torch
from torch.functional import Tensor
import time

class RobotArm2d():
    """2D PRRR robot arm from ardizzone et al. (prismatic, rotational, rotational, rotational)
    Arm can have more or less rotational joints.
    """
    def __init__(self, lengths: list = [0.5, 0.5, 1], sigmas: list = [0.25, 0.5, 0.5, 0.5]):
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.sigmas = Tensor(sigmas)
        self.num_joints = self.sigmas.shape[0]
        self.lengths = Tensor(lengths)
        self.rangex = (-self.lengths.sum()*0.5, self.lengths.sum()*1.2)  # (-0.35, 2.25)
        self.rangey = (-(self.lengths.sum() + self.sigmas[0]), (self.lengths.sum() + self.sigmas[0]))  # (-1.3, 1.3)
        cmap = plt.cm.tab20c
        self.colors = [[cmap(4*c_index + i) for i in range(self.lengths.shape[0])] for c_index in range(5)][-1]
        self.out_dir = "data"
        self.viz_dir = "visualizations"
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        if not os.path.isdir(self.viz_dir):
            os.makedirs(self.viz_dir, exist_ok=True)

    def sample_priors(self, batch_size: int = 1) -> torch.FloatTensor:
        """Normal distributed values of the joint parameters"""
        return torch.randn(batch_size, self.num_joints, device=self.device) * self.sigmas

    def advance_joint(self, current_pos: torch.FloatTensor, length: float, angle: torch.FloatTensor) -> (torch.FloatTensor, torch.FloatTensor):
        """Calculate position of next joint

        :param current_pos: Current position at joint in 2d, size: (n, 2)
        :param length: Length of the arm from current to next position, float
        :param angle: Angle around current joint, (n)
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

        :param thetas: Joint parameters: (n, self.num_joints)
        :return: End effector position in 2d: (n, 2)
        """
        angle = torch.zeros_like(thetas[:, 1], device=self.device)
        p_next = torch.stack([torch.zeros((thetas.shape[0]), device=thetas.device), thetas[:, 0]], axis=1)
        for joint in range(self.num_joints -1):
            angle += thetas[:, joint + 1]
            _, p_next = self.advance_joint(p_next, self.lengths[joint], angle)
        return p_next

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

    def inverse(self, pos: torch.FloatTensor, inverses_each: float = 10, epsilon: float = 5e-2, mult: float = 100) -> torch.FloatTensor:
        """Inverse kinematics with rejection sampling

        :param pos: Target end effector position, size (n, 2)
        :param inverses_each: How many inverses per target end effector
        :param epsilon: Tolerance for prediction compared to ground truth, float
        :param mult: Multiplier how many more samples will be generated each iteration than needed
        :return: Joint parameters, size (n*inverses_each, sigmas.shape[0])
        """
        n = pos.shape[0]
        pdist = torch.nn.PairwiseDistance(p=2)
        thetas = torch.zeros((n*inverses_each, self.sigmas.shape[0]), device=self.device)
        
        # Loop over each target position
        for i in range(n):
            num_thetas_close = 0
            while num_thetas_close < num_inverse_each:
                # Sample thetas, forward, keep thetas with position close to target
                thetas_gen = self.sample_priors(mult*inverses_each)
                pos_gen = self.forward(thetas_gen)
                thetas_close = thetas_gen[pdist(pos[i], pos_gen) <= epsilon]
                num_thetas_close_batch = thetas_close.shape[0]
                if num_thetas_close_batch == 0:
                    continue
                # Fill with thetas that yield a close position
                if num_thetas_close + num_thetas_close_batch > inverses_each:
                    num_diff = inverses_each - num_thetas_close
                    thetas[i*inverses_each + num_thetas_close : (i+1)*inverses_each] = thetas_close[0:num_diff]
                    break
                elif num_thetas_close_batch > 0:
                    thetas[i*inverses_each + num_thetas_close : i*inverses_each + num_thetas_close + num_thetas_close_batch] = thetas_close[:]
                num_thetas_close += num_thetas_close_batch
        return thetas


    def init_plot(self) -> plt.figure:
        """Initialize matplotlib figure"""
        return plt.figure(figsize=(8, 8))

    def viz_forward(self, pos: torch.FloatTensor) -> None:
        """Visualization of forward kinematics positions

        :param pos: End effector position in 2d: (n, 2)
        """
        # Bring pos on cpu for plotting
        pos = pos.cpu().numpy()
        
        fig = self.init_plot()
        plt.scatter(pos[:, 0], pos[:, 1], s=5, alpha=0.5)
        plt.xlim(*self.rangex)
        plt.ylim(*self.rangey)
        plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
        plt.title(f"Forward Kinematics with {pos.shape[0]} samples")
        #TODO: implement saving routine
        fig.savefig(os.path.join(self.viz_dir, "0000_forward") + ".png")
        plt.show()

    def viz_inverse(self, pos: torch.FloatTensor, thetas: torch.FloatTensor, save: bool = True, show: bool = False, fig_name: str = "fig", viz_format: tuple = (".png", ".svg")) -> None:
        """Visualization of inverse kinematic configurations for end effector position

        :param pos: End effector position, size (n, 2), use n=1 to get informative plots
        :param thetas: Joint parameters, size (n, self.num_joints)
        :param save: Bool, True if plot should be saved
        :param show: Bool, True if plot should be displayed
        :param fig_name: Name of the figure without ending or directory, e.g. "fig1"
        :param viz_format: Formats in which the plot should be saved, e.g. (".png", ".svg") or ("png",)
        """

        # Setup plot
        fig = self.init_plot()
        opts = {'alpha': 0.05, 'scale': 1, 'angles': 'xy', 'scale_units': 'xy', 'headlength': 0, 'headaxislength': 0, 'linewidth': 1.0, 'rasterized': True}

        angle = torch.zeros_like(thetas[:, 1])
        p_next = torch.stack([torch.zeros((thetas.shape[0]), device=thetas.device), thetas[:, 0]], axis=1)

        for joint in range(self.num_joints -1):
            # Advance one joint
            angle += thetas[:, joint + 1]
            p_current, p_next = self.advance_joint(p_next, self.lengths[joint], angle)
            # Plot arm between the two positions
            plt.quiver(p_current[:, 0], p_current[:, 1], (p_next-p_current)[:, 0], (p_next-p_current)[:, 1], **{'color': self.colors[joint], **opts})


        # Calculate distance from target
        distance = self.distance_euclidean(pos, p_next)

        # Plot cross to mark end effector position
        l_cross = 0.6
        plt.vlines(pos[:, 0], pos[:, 1]-l_cross, pos[:, 1]+l_cross, ls='-', colors='black', linewidth=.5, alpha=.5, zorder=-1)
        plt.hlines(pos[:, 1], pos[:, 0]-l_cross, pos[:, 0]+l_cross, ls='-', colors='black', linewidth=.5, alpha=.5, zorder=-1)

        plt.xlim(*self.rangex)
        plt.ylim(*self.rangey)
        plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
        # Euclidean position is only calculated to the first entry of pos, while target crosses for all will be displayed
        plt.title(f"Inverse Kinematics with {thetas.shape[0]} samples, mean euc. distance = {distance:.3f}")

        if save:
            for format in viz_format:
                fig.savefig(os.path.join(self.viz_dir, fig_name) + format)
        if show:
            plt.show()
        plt.close(fig)

    def generate_data(self, thetas: torch.FloatTensor, num_inverses: int) -> None:
        """Generate training data: for each prior the end effector position is
        Calculated and for each end effector position num_inverses will be sampled

        :param thetas: Joint parameters, size (n, self.num_joints)
        :param num_inverses: Amount of inverses to be created per end effector position
        """
        pos = self.forward(thetas)
        num_positions = thetas.shape[0]
        guesses = torch.zeros((num_positions * num_inverses, self.num_joints))
        # TODO: finish after implementing inverse kinematics
        raise NotImplementedError


if __name__ == "__main__":
    #TODO: move this to a test class

    ## 5 DOF P RRR R joints
    arm = RobotArm2d(lengths = [0.5, 0.5, 1, 1, 1, 1], sigmas = [0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # num_forward = 10
    # num_inverse_each = 100
    # Viz forward
    # priors = arm.sample_priors(num_forward)
    # pos = arm.forward(priors)
    # print(pos)
    # arm.viz_forward(pos)
    # # Viz and test inverse
    # guesses = arm.inverse(pos, arm.sample_priors(num_inverse_each))
    # arm.viz_inverse(pos, guesses)

    # Test inverse
    start = time.time()
    num_forward = 2
    num_inverse_each = 100
    priors = arm.sample_priors(num_forward)
    pos = arm.forward(priors)
    arm.viz_forward(pos)
    thetas_gen = arm.inverse(pos, num_inverse_each)
    time_taken = time.time() - start
    print(f"time: {time_taken}")
    arm.viz_inverse(pos[0:1], thetas_gen[0:num_inverse_each], fig_name="0000_inverse")

    # ## 4 DOF
    # arm = RobotArm2d()
    # num_forward = 10
    # num_inverse_each = 100
    # # Viz forward
    # # priors = arm.sample_priors(num_forward)
    # # pos = arm.forward(priors)
    # # arm.viz_forward(pos)
    # # # Viz and test inverse
    # # guesses = arm.inverse(pos, arm.sample_priors(num_inverse_each))
    # # arm.viz_inverse(pos, guesses)

    # # Test inverse
    # start = time.time()
    # num_forward = 100
    # num_inverse_each = 1000
    # priors = arm.sample_priors(num_forward)
    # pos = arm.forward(priors)
    # thetas_gen = arm.inverse(pos, num_inverse_each)
    # print(thetas_gen.shape)
    # time_taken = time.time() - start
    # print(f"time: {time_taken}")
    # #print(thetas_gen)
    # arm.viz_inverse(pos[0:1], thetas_gen[0:num_inverse_each])