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

def test_dataset():
    """Test the InverseDataset3d
    Call from main and analyze one position with all its corresponding thetas."""
    path = "data/inverse_data_JustinArm07_1000_100.pickle"
    inv_dataset_3d = InverseDataset3d(path)
    print(len(inv_dataset_3d))
    print(inv_dataset_3d.thetas.shape[0], inv_dataset_3d.pos.shape[0])#rot.shape

    from rokin.Robots import JustinArm07
    robot = JustinArm07()
    sample_idx = np.random.randint(inv_dataset_3d.pos.shape[0], size=(1,)).item()

    forward_frames = robot.get_frames(inv_dataset_3d.thetas[sample_idx*inv_dataset_3d.ratio : (sample_idx + 1) * inv_dataset_3d.ratio].cpu().numpy())
    forward_pos = forward_frames[:, -1, 0:3, 3]
    target_pos = torch.unsqueeze(inv_dataset_3d.pos[sample_idx], 0).cpu().numpy()

    print(forward_pos.shape, target_pos.shape)
    pdist = np.linalg.norm(target_pos-forward_pos, axis=1)
    print(np.max(pdist), np.min(pdist), np.sum(pdist)/pdist.shape[0])
    #TODO: add test for rotation