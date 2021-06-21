from FrEIA.framework import SequenceINN
from matplotlib import pyplot as plt
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

from inn.model import create_inn
from inn.dataset import load_dataset


# TODO deduplicate
def load_model(thetas_dim: int, checkpoint_path: str = "very-long-log/125_checkpoint.pt") -> SequenceINN:
    inn = create_inn(thetas_dim)
    checkpoint = torch.load(checkpoint_path)
    inn.load_state_dict(checkpoint["model"])
    inn.eval()
    return inn


def plot_z(z: torch.tensor):
    """
    Creates a scatter plot of the given positions.
    :param z: Tensor of shape (n, 2)
    """
    plt.figure(figsize=(8.2, 12))
    plt.scatter(z[:, 0], z[:, 1], s=5, alpha=0.1)
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.show()


def plot_predicted_z_distribution(batch_size: int = 128):
    test_dataset, thetas_dim, position_dim = load_dataset("data/test.pickle")
    test_loader = DataLoader(test_dataset, batch_size, num_workers=2)

    inn = load_model(thetas_dim)

    # required for the list comprehension to work
    torch.multiprocessing.set_sharing_strategy('file_system')
    with torch.no_grad():
        z_pred = [inn(thetas)[0] for thetas, _ in test_loader]
    z_pred = torch.cat(z_pred, dim=0)[:, :2]

    print(z_pred.size())
    plot_z(z_pred)


def plot_ground_truth_z_distribution():
    noise_batch = torch.randn(1000000, 2)
    plot_z(noise_batch)


if __name__ == "__main__":
    plot_predicted_z_distribution()
    plot_ground_truth_z_distribution()
