import pickle

from FrEIA.framework import SequenceINN
import torch

from inn.mmd import forward_mmd
from inn.model import create_inn


BATCH_SIZE = 10000
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

    train_priors = torch.tensor(train_data["priors"], dtype=torch.float)[:BATCH_SIZE, :]
    train_positions = torch.tensor(train_data["positions"], dtype=torch.float)[:BATCH_SIZE, :]
    test_priors = torch.tensor(test_data["priors"], dtype=torch.float)[:BATCH_SIZE, :]
    test_positions = torch.tensor(test_data["positions"], dtype=torch.float)[:BATCH_SIZE, :]

    # Forward
    ##########

    with torch.no_grad():
        predicted_positions, _ = inn(train_priors)
        predicted_positions = predicted_positions[:, Z_DIM:]

    print("Forward:")
    print("MMD between predicted positions and test set positions: ", end="")
    print(torch.mean(forward_mmd(predicted_positions, test_positions)).item())
    print("MMD between train set positions and test set positions: ", end="")
    print(torch.mean(forward_mmd(train_positions, test_positions)).item())

    # Backward
    ###########

    noise_batch = torch.randn(BATCH_SIZE, Z_DIM)
    train_positions_with_noise = torch.cat((noise_batch, train_positions), dim=1)
    with torch.no_grad():
        predicted_priors, _ = inn(train_positions_with_noise, rev=True)

    print("Backward:")
    print("MMD between predicted priors and test set priors: ", end="")
    print(torch.mean(forward_mmd(predicted_priors, test_priors)).item())
    print("MMD between train set priors and test set priors: ", end="")
    print(torch.mean(forward_mmd(train_priors, test_priors)).item())


if __name__ == "__main__":
    evaluate()
