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
    """Test the InverseDataset2d"""
    path = "data/inverse.pickle"
    inv_dataset_2d = InverseDataset2d(path)
    print(len(inv_dataset_2d))

    arm = RobotArm2d()
    for i in range(10):
        sample_idx = torch.randint(len(inv_dataset_2d), size=(1,)).item()
        thetas, position = inv_dataset_2d[sample_idx]
        print(thetas, 50*'#', position)
        thetas = torch.unsqueeze(thetas, 0).numpy()
        position = torch.unsqueeze(position, 0).numpy()
        print(thetas.shape, position.shape)
        arm.viz_inverse(position, thetas)
