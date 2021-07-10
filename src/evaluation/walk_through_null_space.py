import pickle

from FrEIA.framework import SequenceINN
import torch

from inn.model import create_inn
from kinematics.robot_arm_2d_torch import RobotArm2d


PRIORS_DIM = 4


# TODO deduplicate
def _load_model(checkpoint_path: str = "very-long-log/125_checkpoint.pt") -> SequenceINN:
    inn = create_inn(PRIORS_DIM)
    checkpoint = torch.load(checkpoint_path)
    inn.load_state_dict(checkpoint["model"])
    inn.eval()
    return inn


def walk_through_null_space(num_samples: int = 32):
    # only use one position
    with open("data/inverse1.pickle", "rb") as file:
        data = pickle.load(file)
    position = torch.tensor(data["positions"], dtype=torch.float)
    positions = position.repeat(num_samples, 1)

    # use evenly spaced numbers as latent variables instead of noise
    latent_vars = torch.linspace(-1.5, 1.5, steps=num_samples)
    latent_vars = latent_vars.repeat(2, 1).T
    positions_with_latent = torch.cat((latent_vars, positions), dim=1)

    inn = _load_model()

    with torch.no_grad():
        thetas_pred, _ = inn(positions_with_latent, rev=True)

    # visualise results
    robot_arm = RobotArm2d()

    thetas = torch.tensor(data["posteriors"], dtype=torch.float)[:32, :]
    print(positions.size())
    print(thetas.size())
    robot_arm.viz_inverse(positions, thetas_pred, save=False, show=True)
