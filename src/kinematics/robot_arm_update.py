import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

class RobotArm2d():
    """2D PRRR robot arm from ardizzone et al."""
    def __init__(self, lengths=[0.5, 0.5, 1], sigmas=[0.25, 0.5, 0.5, 0.5]):
        self.sigmas = np.array(sigmas)
        self.lengths = np.array(lengths)
    
    def forward(self, parameters):
        t1, t2, t3, t4 = parameters
        l1, l2, l3 = self.lengths
        # TODO: investigate why ardizzone took different angle directions (signs in sin and cos)
        x = l1 * np.cos(t2) + l2 * np.cos(t3 + t2) + l3 * np.cos(t4 + t3 + t2)
        y = t1 + l1 * np.sin(t2) + l2 * np.sin(t3 + t2) + l3 * np.sin(t4 + t3 + t2)
        return np.array([x, y])

    def sample_priors(self):
        return np.random.randn(4) * self.sigmas

    def inverse(self, pos_tip, guess, epsilon=1e-2, max_iter=3000, lr=0.01):
        pos_tip = np.array(pos_tip)
        guess = np.array(guess)
        l1, l2, l3 = self.lengths
        # Gradient descent
        iter = 0
        for iter in range(max_iter):
            t1, t2, t3, t4 = guess

            # Jacobian
            J = np.array([[0, - l1 * np.sin(t2) - l2 * np.sin(t2 + t3) - l3 * np.sin(t2 + t3 + t4), - l2 * np.sin(t2 + t3) - l3 * np.sin(t2 + t3 + t4), - l3 * np.sin(t2 + t3 + t4)],
                        [1, l1 * np.cos(t2) + l2 * np.cos(t2 + t3) + l3 * np.cos(t2 + t3 + t4), l2 * np.cos(t2 + t3) + l3 * np.cos(t2 + t3 + t4),  l3 * np.cos(t2 + t3 + t4)]])
            forward = self.forward(guess)
            error = pos_tip - forward
            #print(error)
            guess = guess + lr * np.dot(J.T, error)
            
            if (np.linalg.norm(error) < epsilon):
                break
            #print(guess)
        print(iter)
        print(f"Error: {np.linalg.norm(error)}")
        return guess

    def viz_forward(self):
        pass
    def viz_inverse(self):
        pass
    def save(self):
        pass


if __name__ == "__main__":
    arm = RobotArm2d()

    # Viz forward
    tip_array = np.zeros((1000,2))
    for i in range(1000):
        tip_array[i] = arm.forward(arm.sample_priors())
    plt.scatter(tip_array[:,0], tip_array[:,1])
    plt.show()

    # Viz and test inverse
    theta = arm.sample_priors()
    pos_for = np.array(arm.forward(theta))
    inv = arm.inverse(pos_for, np.array([0.25, 0.5, 0.5, 0.5]))
    pos_inv = arm.forward(inv)
    print(100 * '#')

    pts = np.array([pos_for, pos_inv])
    pts = np.concatenate((pts, np.array([[0,0], [1,0]])))
    plt.scatter(pts[:,0], pts[:,1])
    plt.show()
    print(theta)
    print(inv)