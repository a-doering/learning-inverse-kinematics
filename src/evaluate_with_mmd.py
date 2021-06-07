import pickle

from FrEIA.framework import SequenceINN
import torch

from inn.mmd import forward_mmd
from inn.model import create_inn


PRIORS_DIM = 4
Z_DIM = 2


def load_model(checkpoint_path: str = "very-long-log/125_checkpoint.pt") -> SequenceINN:
    inn = create_inn(PRIORS_DIM)
    checkpoint = torch.load(checkpoint_path)
    inn.load_state_dict(checkpoint["model"])
    inn.eval()
    return inn


def evaluate():
    with open("data/train.pickle", "rb") as train_file:
        train_data = pickle.load(train_file)
    with open("data/test.pickle", "rb") as test_file:
        test_data = pickle.load(test_file)

    inn = load_model()

    train_priors = torch.tensor(train_data["priors"], dtype=torch.float)[:10000, :]
    train_positions = torch.tensor(train_data["positions"], dtype=torch.float)[:10000, :]
    with torch.no_grad():
        predicted_positions, _ = inn(train_priors)
        predicted_positions = predicted_positions[:, Z_DIM:]

    test_positions = torch.tensor(test_data["positions"], dtype=torch.float)[:10000, :]

    print("MMD between predicted positions and test set positions: ", end="")
    print(torch.mean(forward_mmd(predicted_positions, test_positions)).item())
    print("MMD between train set positions and test seet positions: ", end="")
    print(torch.mean(forward_mmd(train_positions, test_positions)).item())


if __name__ == "__main__":
    evaluate()
