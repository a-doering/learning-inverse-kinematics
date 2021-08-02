import pickle
from torch.utils.data import Dataset
import torch
import numpy as np


class InverseDataset3d(Dataset):

    def __init__(self, path: str, use_rot: bool = False):
        self.path = path
        with open(self.path, "rb") as file:
            data = pickle.load(file)
        self.thetas = torch.tensor(data["thetas"], dtype=torch.float)
        self.pos = torch.tensor(data["pos"], dtype=torch.float)
        self.use_rot = use_rot
        if use_rot:
            self.rot = torch.tensor(data["rot"], dtype=torch.float)
        self.ratio = self.thetas.shape[0] // self.pos.shape[0]

    def __len__(self) -> int:
        return self.thetas.shape[0]

    def __getitem__(self, index: int):
        if self.use_rot:
            return self.thetas[index], self.pos[index // self.ratio], self.rot[index // self.ratio]
        else:
            return self.thetas[index], self.pos[index // self.ratio]

#TODO: update test to 3d
