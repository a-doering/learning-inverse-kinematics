import pickle

import torch
from torch.utils.data import TensorDataset


def load_data(path: str = "data/forward.pickle") -> TensorDataset:
    with open(path, "rb") as file:
        data = pickle.load(file)

    thetas = torch.tensor(data["thetas"], dtype=torch.float)
    positions = torch.tensor(data["positions"], dtype=torch.float)

    return TensorDataset(thetas, positions)
