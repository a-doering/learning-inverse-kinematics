import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time


class RobotArm2d():
    """2D PRRR robot arm from ardizzone et al. (prismatic, rotational, rotational, rotational"""
    def __init__(self, lengths=[0.5, 0.5, 1], sigmas=[0.25, 0.5, 0.5, 0.5]):
        self.sigmas = np.array(sigmas)
        self.lengths = np.array(lengths)
        self.rangex = (-0.5, 2.5)  # (-0.35, 2.25)
        self.rangey = (-1.5, 1.5)  # (-1.3, 1.3)
        cmap = cm.tab20c
        self.colors = [[cmap(4*c_index), cmap(4*c_index+1), cmap(4*c_index+2)] for c_index in range(5)][-1]
        self.out_dir = "data"

    def forward(self, parameters: np.array) -> np.array:
        """Forward kinematics of given joint configuration

        :param parameters: Joint parameters: distance, theta2, theta3, theta4
        :return: End effector position in 2d as np.array
        """
        t1, t2, t3, t4 = parameters
        l1, l2, l3 = self.lengths
        # TODO: investigate why ardizzone took different angle directions (signs in sin and cos)
        x = l1 * np.cos(t2) + l2 * np.cos(t3 + t2) + l3 * np.cos(t4 + t3 + t2)
        y = t1 + l1 * np.sin(t2) + l2 * np.sin(t3 + t2) + l3 * np.sin(t4 + t3 + t2)
        return np.array([x, y])

    def sample_priors(self, samples=1):
        """Normal distributed values of the joint parameters"""
        if samples == 1:
            return np.random.randn(4) * self.sigmas
        else:
            return np.random.randn(samples, 4) * self.sigmas

    def inverse(self, tcp_pos: np.array, guess: np.array, epsilon=5e-2, max_steps=3000, lr=0.2) -> np.array:
        """Inverse kinematics with gradient descent

        :param tcp_pos: Position of the tool center point of the end effector as list or np.array of size (2)
        :param guess: Inital guess for the joint parameters, list or np.array of size (4)
        :param epsilon: Tolerance for prediction compared to ground truth, float
        :param max_steps: Maximum number of steps for gradient descent
        :param lr: Learn rate, float
        :return: Joint parameters as np.array of size (4)
        """
        tcp_pos = np.array(tcp_pos)
        guess = np.array(guess)
        l1, l2, l3 = self.lengths
        # Gradient descent
        steps = 0
        for steps in range(max_steps):
            t1, t2, t3, t4 = guess

            # Jacobian
            J = np.array([[0, - l1 * np.sin(t2) - l2 * np.sin(t2 + t3) - l3 * np.sin(t2 + t3 + t4), - l2 * np.sin(t2 + t3) - l3 * np.sin(t2 + t3 + t4), - l3 * np.sin(t2 + t3 + t4)],
                         [1, l1 * np.cos(t2) + l2 * np.cos(t2 + t3) + l3 * np.cos(t2 + t3 + t4), l2 * np.cos(t2 + t3) + l3 * np.cos(t2 + t3 + t4),  l3 * np.cos(t2 + t3 + t4)]])
            forward = self.forward(guess)
            error = tcp_pos - forward
            guess = guess + lr * np.dot(J.T, error)
            if (np.linalg.norm(error) < epsilon):
                break
        # print(f"Error: {np.linalg.norm(error)}, number of steps: {steps}")
        return guess

    def viz_forward(self, tcp_array: np.array, priors=None) -> None:
        """Visualization of forward kinematics positions

        :param tcp_array: Tool center point np.array of size (n, 2)
        :param priors: Joint priors corresponding to tcp_array of size (n, 4)
        """
        fig = self.init_plot()
        if priors is not None:
            # TODO: implement plotting of arm joints
            pass
        else:
            # Plotting of tcp positions
            plt.scatter(tcp_array[:, 0], tcp_array[:, 1], s=5)
        plt.xlim(*self.rangex)
        plt.ylim(*self.rangey)
        plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
        plt.title(f"Forward Kinematics with {tcp_array.shape[0]} samples")
        plt.show()

    def init_plot(self) -> plt.figure:
        """Initialize matplotlib figure"""
        return plt.figure(figsize=(8, 8))

    def viz_inverse(self, tcp_pos: np.array, guesses_array: np.array) -> None:
        """Visualization of inverse kinematic configurations for one tcp position with all arms

        :param tcp_pos: Tool center point position, np.array of size (2)
        :param guesses_array: Joint configurations that lead to tcp_pos, np.array of size (n, 4)
        """
        l1, l2, l3 = self.lengths
        starting_pos = np.zeros((guesses_array.shape[0], 2))
        starting_pos[:, 1] = guesses_array[:, 0]

        p0 = np.zeros((guesses_array.shape[0], 2))
        p0[:, 1] = guesses_array[:, 0]

        angle = np.array(guesses_array[:, 1])
        p1 = np.array(p0)
        p1[:, 0] += l1 * np.cos(angle)
        p1[:, 1] += l1 * np.sin(angle)

        angle = angle + guesses_array[:, 2]
        p2 = np.array(p1)
        p2[:, 0] += l2 * np.cos(angle)
        p2[:, 1] += l2 * np.sin(angle)

        angle = angle + guesses_array[:, 3]
        p3 = np.array(p2)
        p3[:, 0] += l3 * np.cos(angle)
        p3[:, 1] += l3 * np.sin(angle)

        fig = self.init_plot()
        opts = {'alpha': 0.05, 'scale': 1, 'angles': 'xy', 'scale_units': 'xy', 'headlength': 0, 'headaxislength': 0, 'linewidth': 1.0, 'rasterized': True}
        plt.quiver(p0[:, 0], p0[:, 1], (p1-p0)[:, 0], (p1-p0)[:, 1], **{'color': self.colors[0], **opts})
        plt.quiver(p1[:, 0], p1[:, 1], (p2-p1)[:, 0], (p2-p1)[:, 1], **{'color': self.colors[1], **opts})
        plt.quiver(p2[:, 0], p2[:, 1], (p3-p2)[:, 0], (p3-p2)[:, 1], **{'color': self.colors[2], **opts})
        plt.xlim(*self.rangex)
        plt.ylim(*self.rangey)
        plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
        plt.title(f"Inverse Kinematics with {guesses_array.shape[0]} samples")
        plt.show()

    def save_forward(self, tcp_array: np.array, priors: np.array) -> None:
        """Saving of forward kinematics

        :param tcp_array: Tool center point np.array of size (n, 2)
        :param priors: Joint priors corresponding to tcp_array of size (n, 4)
        """
        data = {
            "positions": tcp_array,
            "priors": priors
        }
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.out_dir + os.path.sep + "forward.pickle", "wb") as file:
            pickle.dump(data, file)

    def save_inverse(self, tcp_pos: np.array, guesses_array: np.array) -> None:
        """Saving of inverse kinematics

        :param tcp_pos: Tool center point np.array of size (2) or (n, 2)
        :param guesses_array: Joint configurations that lead to tcp_pos, np.array of size (m, 4) or (n*M, 4)
        """
        data = {
            "positions": tcp_pos,
            "posteriors": guesses_array
        }
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.out_dir + os.path.sep + "inverse.pickle", "wb") as file:
            pickle.dump(data, file)

    def generate_data(self, priors: np.array, num_inverses: int) -> None:
        """Generating training data: for each prior the tcp position will be
        Calculated and for each tcp position num_inverses will be sampled

        :param priors: Joint priors np.array of size (n, 4)
        :param num_inverses: Amount of inverses to be created per tcp_position
        """
        # Calculating forward kinematics
        if priors.ndim == 1:
            num_tcp = 1
        else:
            num_tcp = priors.shape[0]
        tcp_array = np.zeros((num_tcp, 2))
        for i in range(num_tcp):
            tcp_array[i] = self.forward(priors[i])

        # Calculating inverse kinematics
        guesses_array = np.zeros((num_tcp * num_inverses, 4))
        start = time.time()
        for i in range(num_tcp):
            for j in range(num_inverses):
                guesses_array[i * num_inverses + j] = arm.inverse(tcp_array[i], arm.sample_priors())
        end = time.time()
        print(f"Time taken: {end-start} seconds")

        # Saving
        self.save_inverse(tcp_array, guesses_array)

if __name__ == "__main__":
    arm = RobotArm2d()
    # Samples to be generated
    num_forward = 100
    num_inverses_each = 1000
    arm.generate_data(arm.sample_priors(num_forward), num_inverses_each)

    # # Viz forward
    # tcp_array = np.zeros((n, 2))
    # priors = arm.sample_priors(n)
    # for i in range(n):
    #     tcp_array[i] = arm.forward(priors[i])
    # arm.save_forward(tcp_array, priors)
    # arm.viz_forward(tcp_array)

    # # Viz and test inverse
    # theta = arm.sample_priors()
    # tcp_pos = np.array(arm.forward(theta))
    # start = time.time()
    # guesses_array = np.zeros((n, 4))
    # for i in range(n):
    #     guesses_array[i] = arm.inverse(tcp_pos, arm.sample_priors())
    # print(time.time()-start)
    # arm.save_inverse(tcp_pos, guesses_array)
    # arm.viz_inverse(tcp_pos, guesses_array)
