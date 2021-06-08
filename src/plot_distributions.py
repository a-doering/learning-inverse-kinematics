from FrEIA.framework import SequenceINN
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from inn.model import create_inn
from inn.dataset import load_dataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


RANGE_X = (-0.5, 2.5)
RANGE_Y = (-1.5, 1.5)


# TODO deduplicate
def load_model(priors_dim: int, checkpoint_path: str = "very-long-log/125_checkpoint.pt") -> SequenceINN:
    inn = create_inn(priors_dim)
    checkpoint = torch.load(checkpoint_path)
    inn.load_state_dict(checkpoint["model"])
    print(checkpoint["epoch"])
    inn.eval()
    return inn


def plot_positions(positions: torch.tensor):
    """
    Creates a scatter plot of the given positions.
    :param positions: Tensor of shape (n, 2)
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], s=5)
    plt.xlim(*RANGE_X)
    plt.ylim(*RANGE_Y)
    plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
    plt.show()


def plot_predicted_position_distribution(batch_size: int = 128):
    test_dataset, priors_dim, position_dim = load_dataset("data/test.pickle")
    test_loader = DataLoader(test_dataset, batch_size, num_workers=2)

    inn = load_model(priors_dim)

    with torch.no_grad():
        positions_pred = [inn(priors)[0] for priors, _ in test_loader]
    positions_pred = torch.cat(positions_pred, dim=0)[:, -position_dim:]

    print(positions_pred.size())
    plot_positions(positions_pred)


if __name__ == "__main__":
    plot_predicted_position_distribution()
