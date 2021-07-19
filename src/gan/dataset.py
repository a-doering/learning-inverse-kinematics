import pickle
from torch.utils.data import Dataset
import torch
from kinematics.robot_arm_2d_torch import RobotArm2d


class InverseDataset2d(Dataset):

    def __init__(self, path: str):
        self.path = path
        with open(self.path, "rb") as file:
            data = pickle.load(file)
        self.thetas = torch.tensor(data["thetas"], dtype=torch.float)
        self.pos = torch.tensor(data["pos"], dtype=torch.float)
        self.ratio = self.thetas.shape[0] // self.pos.shape[0]

    def __len__(self) -> int:
        return self.thetas.shape[0]

    def __getitem__(self, index: int):
        return self.thetas[index], self.pos[index // self.ratio]


def test_dataset():
    """Test the InverseDataset2d
    Call from main and create a plot for one position with all its corresponding thetas."""
    path = "data/inverse_data_7_1000_100.pickle"
    inv_dataset_2d = InverseDataset2d(path)
    print(len(inv_dataset_2d))
    arm = RobotArm2d(lengths = [0.5, 0.5, 1, 1, 1, 1], sigmas = [0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    print(inv_dataset_2d.thetas.shape[0], inv_dataset_2d.pos.shape[0])
    sample_idx = torch.randint(inv_dataset_2d.pos.shape[0], size=(1,)).item()
    arm.viz_inverse(torch.unsqueeze(inv_dataset_2d.pos[sample_idx].cpu(),0), inv_dataset_2d.thetas[sample_idx*inv_dataset_2d.ratio : (sample_idx + 1) * inv_dataset_2d.ratio].cpu())
