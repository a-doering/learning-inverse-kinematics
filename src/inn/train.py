from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import create_inn


def train(
    batch_size: int = 50,
    lr: float = 1e-4,
    epochs: int = 100000,
):
    inn = create_inn()
    optimizer = Adam(inn.parameters(), lr=lr)
    train_loader = DataLoader(load_dataset(), batch_size, shuffle=True, num_workers=2, drop_last=False)

    for epoch in range(epochs):
        loss_sum = 0

        for thetas, positions in train_loader:
            # thetas is what is called "x" in the Ardizzone et al. paper, and positions is "y"

            optimizer.zero_grad()

            # the INN also returns the log Jacobian determinant, but this isn't used here
            positions_pred, _ = inn(thetas)
            loss = F.mse_loss(positions_pred, positions)

            loss.backward()
            optimizer.step()
            loss_sum += loss

        loss_mean = loss_sum / len(train_loader)
        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Train loss: {loss_mean}")


if __name__ == "__main__":
    train()
