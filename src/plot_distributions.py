import pickle
from FrEIA.framework import SequenceINN
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

from inn.model import create_inn
from inn.dataset import load_dataset


RANGE_X = (-2.0, 2.2)
RANGE_Y = (-3.0, 3.0)


# TODO deduplicate
def load_model(thetas_dim: int, checkpoint_path: str = "very-long-log/125_checkpoint.pt") -> SequenceINN:
    inn = create_inn(thetas_dim)
    checkpoint = torch.load(checkpoint_path)
    inn.load_state_dict(checkpoint["model"])
    inn.eval()
    return inn


def plot_positions(positions: torch.tensor):
    """
    Creates a scatter plot of the given positions.
    :param positions: Tensor of shape (n, 2)
    """
    plt.figure(figsize=(8.2, 12))
    plt.scatter(positions[:, 0], positions[:, 1], s=5, alpha=0.1)
    plt.xlim(*RANGE_X)
    plt.ylim(*RANGE_Y)
    plt.axvline(x=0, ls=':', c='gray', linewidth=.5)
    plt.show()


def plot_predicted_position_distribution(batch_size: int = 128):
    test_dataset, thetas_dim, position_dim = load_dataset("data/test.pickle")
    test_loader = DataLoader(test_dataset, batch_size, num_workers=2)

    inn = load_model(thetas_dim)

    # required for the list comprehension to work
    torch.multiprocessing.set_sharing_strategy('file_system')
    with torch.no_grad():
        positions_pred = [inn(thetas)[0] for thetas, _ in test_loader]
    positions_pred = torch.cat(positions_pred, dim=0)[:, -position_dim:]

    print(positions_pred.size())
    plot_positions(positions_pred)


def plot_ground_truth_position_distribution():
    with open("data/test.pickle", "rb") as test_file:
        test_data = pickle.load(test_file)
    print(test_data["positions"].shape)
    plot_positions(test_data["positions"])


def plot_thetas(batch_size: int = 128):
    # Ground truth thetas
    ######################

    with open("data/test.pickle", "rb") as file:
        data = pickle.load(file)
    ground_truth_thetas = data["priors"]

    scaler = StandardScaler()
    ground_truth_thetas_scaled = scaler.fit_transform(ground_truth_thetas)
    pca = PCA(n_components=2)
    ground_truth_thetas_2d = pca.fit_transform(ground_truth_thetas_scaled)

    plt.figure(figsize=(8, 8))
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.scatter(ground_truth_thetas_2d[:, 0], ground_truth_thetas_2d[:, 1], s=5, alpha=0.02)
    plt.draw()

    # Predicted thetas
    ###################

    test_dataset, thetas_dim, position_dim = load_dataset("data/test.pickle")
    test_loader = DataLoader(test_dataset, batch_size, num_workers=2)

    inn = load_model(ground_truth_thetas.shape[1])

    def apply_inn(positions_batch):
        noise_batch = torch.randn(positions_batch.size(0), 2)
        positions_with_noise = torch.cat((noise_batch, positions_batch), dim=1)
        thetas, _ = inn(positions_with_noise, rev=True)
        return thetas

    # required for the list comprehension to work
    torch.multiprocessing.set_sharing_strategy('file_system')
    with torch.no_grad():
        thetas_pred = [apply_inn(positions_batch) for _, positions_batch in test_loader]
    thetas_pred = torch.cat(thetas_pred, dim=0)

    thetas_pred_scaled = scaler.transform(thetas_pred)
    thetas_pred_2d = pca.transform(thetas_pred_scaled)

    plt.figure(figsize=(8, 8))
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.scatter(thetas_pred_2d[:, 0], thetas_pred_2d[:, 1], s=5, alpha=0.02)
    plt.show()


def plot_ground_truth_null_space():
    # there should be only one position with many priors
    with open("data/inverse1.pickle", "rb") as file:
        data = pickle.load(file)
    ground_truth_thetas = data["posteriors"]
    positions = torch.tensor(data["positions"], dtype=torch.float)
    positions = positions.repeat(ground_truth_thetas.shape[0], 1)

    scaler = StandardScaler()
    ground_truth_thetas_scaled = scaler.fit_transform(ground_truth_thetas)
    pca = PCA(n_components=2)
    ground_truth_thetas_2d = pca.fit_transform(ground_truth_thetas_scaled)

    inn = load_model(ground_truth_thetas.shape[1])
    noise_batch = torch.randn(positions.shape[0], 2)
    positions_with_noise = torch.cat((noise_batch, positions), dim=1)

    with torch.no_grad():
        thetas_pred, _ = inn(positions_with_noise, rev=True)
    thetas_pred_scaled = scaler.transform(thetas_pred)
    thetas_pred_2d = pca.transform(thetas_pred_scaled)

    plt.figure(figsize=(8, 8))
    plt.scatter(ground_truth_thetas_2d[:, 0], ground_truth_thetas_2d[:, 1], s=5, alpha=0.2)
    plt.draw()

    plt.figure(figsize=(8, 8))
    plt.scatter(thetas_pred_2d[:, 0], thetas_pred_2d[:, 1], s=5, alpha=0.2)
    plt.show()


if __name__ == "__main__":
    plot_thetas()
