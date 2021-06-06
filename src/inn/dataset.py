import pickle
from typing import Tuple

import torch
from torch.utils.data import TensorDataset


def load_dataset(path: str = "data/forward.pickle") -> Tuple[TensorDataset, int, int]:
    """
    :param path: Path to the file containing the pickled training data
    :return: 1. A dataset that returns tuples of (priors, positions), each of size (batch_size, 2) and dtype=torch.float
             2. Dimensionality of the priors vectors
             3. Dimensionality of the position vectors
    """
    with open(path, "rb") as file:
        data = pickle.load(file)

    priors = torch.tensor(data["priors"], dtype=torch.float)
    positions = torch.tensor(data["positions"], dtype=torch.float)

    return TensorDataset(priors, positions), priors.size(1), positions.size(1)
