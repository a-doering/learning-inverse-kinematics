import pickle

from FrEIA.framework import SequenceINN
import torch

from inn.mmd import forward_mmd
from inn.model import create_inn


BATCH_SIZE = 10000
PRIORS_DIM = 4
Z_DIM = 2


# TODO deduplicate
def load_model(checkpoint_path: str = "very-long-log/125_checkpoint.pt") -> SequenceINN:
    inn = create_inn(PRIORS_DIM)
    checkpoint = torch.load(checkpoint_path)
    inn.load_state_dict(checkpoint["model"])
    inn.eval()
    return inn


def evaluate():
    with open("data/inverse1.pickle", "rb") as test_file:
        test_data = pickle.load(test_file)

    inn = load_model()

    priors = torch.tensor(test_data["posteriors"], dtype=torch.float)
    positions = torch.tensor(test_data["positions"], dtype=torch.float)
    positions = positions.repeat(BATCH_SIZE, 1)
    priors_a = priors[:BATCH_SIZE, :]
    priors_b = priors[BATCH_SIZE:2*BATCH_SIZE, :]

    noise_batch = torch.randn(BATCH_SIZE, Z_DIM)
    positions_a_with_noise = torch.cat((noise_batch, positions), dim=1)
    with torch.no_grad():
        predicted_priors, _ = inn(positions_a_with_noise, rev=True)

    print("Backward:")
    print("MMD between predicted priors and priors from the ground truth distribution: ", end="")
    print(torch.mean(forward_mmd(predicted_priors, priors_b)).item())
    print("MMD between two samples of priors from the ground truth distribution: ", end="")
    print(torch.mean(forward_mmd(priors_a, priors_b)).item())


if __name__ == "__main__":
    evaluate()
