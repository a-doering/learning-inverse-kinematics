import pickle

import torch
from torch.utils.data import TensorDataset


def load_dataset(path: str = "data/forward.pickle") -> TensorDataset:
    """
    :param path: Path to the file containing the pickled training data
    :return: A dataset that returns tuples of (priors, positions), each of size (batch_size, 2) and dtype=torch.float
    """
    with open(path, "rb") as file:
        data = pickle.load(file)

    priors = torch.tensor(data["priors"], dtype=torch.float)
    positions = torch.tensor(data["positions"], dtype=torch.float)

    return TensorDataset(priors, positions)
