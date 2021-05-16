import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time


class RobotArm2d():
    """2D PRRR robot arm from ardizzone et al."""
    def __init__(self, lengths=[0.5, 0.5, 1], sigmas=[0.25, 0.5, 0.5, 0.5]):
        self.sigmas = np.array(sigmas)
        self.lengths = np.array(lengths)
        self.rangex = (-0.5, 2.5)  # (-0.35, 2.25)
        self.rangey = (-1.5, 1.5)  # (-1.3, 1.3)
        cmap = cm.tab20c
        self.colors = [[cmap(4*c_index), cmap(4*c_index+1), cmap(4*c_index+2)] for c_index in range(5)][-1]

    def forward(self, parameters):
        t1, t2, t3, t4 = parameters
        l1, l2, l3 = self.lengths
        # TODO: investigate why ardizzone took different angle directions (signs in sin and cos)
        x = l1 * np.cos(t2) + l2 * np.cos(t3 + t2) + l3 * np.cos(t4 + t3 + t2)
        y = t1 + l1 * np.sin(t2) + l2 * np.sin(t3 + t2) + l3 * np.sin(t4 + t3 + t2)
        return np.array([x, y])

    def sample_priors(self):
        return np.random.randn(4) * self.sigmas

    def inverse(self, pos_tip, guess, epsilon=5e-2, max_steps=3000, lr=0.2):
        pos_tip = np.array(pos_tip)
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
            error = pos_tip - forward
            guess = guess + lr * np.dot(J.T, error)
            if (np.linalg.norm(error) < epsilon):
                break
        # print(f"Error: {np.linalg.norm(error)}, number of steps: {steps}")
        return guess

    def viz_forward(self, tcp_array: np.array, priors=None) -> None:
        fig = self.init_plot()
        if priors is not None:
            # TODO: implement plotting of arm joints
            pass
        else:
            plt.scatter(tcp_array[:, 0], tcp_array[:, 1], s=5)
        plt.xlim(*self.rangex)
        plt.ylim(*self.rangey)
        plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
        plt.title(f"Forward Kinematics with {tcp_array.shape[0]} samples")
        plt.show()

    def init_plot(self) -> None:
        return plt.figure(figsize=(8, 8))

    def viz_inverse(self, tcp_pos: np.array, guesses_array: np.array) -> None:

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

    def save(self):
        pass


if __name__ == "__main__":
    arm = RobotArm2d()

    # Viz forward
    tip_array = np.zeros((1000, 2))
    for i in range(1000):
        tip_array[i] = arm.forward(arm.sample_priors())
    arm.viz_forward(tip_array)

    # Viz and test inverse
    theta = arm.sample_priors()
    pos_for = np.array(arm.forward(theta))
    start = time.time()
    guesses_array = np.zeros((1000, 4))
    for i in range(1000):
        guesses_array[i] = arm.inverse(pos_for, arm.sample_priors())
    print(time.time()-start)
    arm.viz_inverse(pos_for, guesses_array)
