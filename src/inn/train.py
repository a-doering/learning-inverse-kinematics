import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import load_dataset
from model import create_inn
import mmd


# TODO un-hard-code
# priors dim
X_DIM = 4
# position dim
Y_DIM = 2
# X_DIM == Y_DIM + Z_DIM, therefore Z_DIM = X_DIM - Y_DIM
Z_DIM = X_DIM - Y_DIM

# factors for the loss terms
FORWARD_FIT_FACTOR = 1
FORWARD_MMD_FACTOR = 50
BACKWARD_MMD_FACTOR = 500
RECONSTRUCTION_FACTOR = 1


def loss_forward_mmd(out, y):
    # Shorten output, and remove gradients wrt y, for latent loss
    output_block_grad = torch.cat((out[:, :Z_DIM], out[:, -Y_DIM:].data), dim=1)
    y_short = torch.cat((y[:, :Z_DIM], y[:, -Y_DIM:]), dim=1)

    l_forward_fit = FORWARD_FIT_FACTOR * mmd.l2_fit(out[:, Z_DIM:], y[:, Z_DIM:])
    l_forward_mmd = FORWARD_MMD_FACTOR * torch.mean(mmd.forward_mmd(output_block_grad, y_short))

    return l_forward_fit, l_forward_mmd


def loss_backward_mmd(x, y, inn):
    # TODO use jac=False ?
    x_samples, _ = inn(y, rev=True)
    MMD = mmd.backward_mmd(x, x_samples)
    return BACKWARD_MMD_FACTOR * torch.mean(MMD)


def loss_reconstruction(out_y, y, x, inn):
    cat_inputs = [out_y[:, :Z_DIM], out_y[:, -Y_DIM:]]
    # TODO use jac=False ?
    x_reconstructed, _ = inn(torch.cat(cat_inputs, 1), rev=True)
    return RECONSTRUCTION_FACTOR * mmd.l2_fit(x_reconstructed, x)


def train(
    batch_size: int = 128,
    lr: float = 1e-6,
    epochs: int = 10000000,
    lr_scheduler_patience: int = 10,
    val_set_portion: float = 0.1,  # portion of the dataset that will be used for validation
):
    inn = create_inn(X_DIM)
    optimizer = Adam(inn.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, verbose=True)

    # prepare dataset
    dataset = load_dataset()
    val_set_size = int(len(dataset) * val_set_portion)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_set_size, val_set_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)

    for epoch in range(epochs):

        # Training
        ###########

        train_loss_history = []
        inn.train()

        for priors, positions in train_loader:
            noise_batch = torch.randn(batch_size, Z_DIM)
            positions = torch.cat((noise_batch, positions), dim=1)

            # TODO use jac=False ?
            positions_pred, _ = inn(priors)

            batch_losses = []
            batch_losses.extend(loss_forward_mmd(positions_pred, positions))
            batch_losses.append(loss_backward_mmd(priors, positions, inn))
            batch_losses.append(loss_reconstruction(positions_pred.data, positions, priors, inn))

            train_loss_history.append([batch_loss.item() for batch_loss in batch_losses])

            total_batch_loss = sum(batch_losses)
            total_batch_loss.backward()
            optimizer.step()

        train_loss_mean = np.mean(train_loss_history, axis=0)
        lr_scheduler.step(np.mean(train_loss_mean))
        print(f"[Epoch {epoch}] Train loss: {np.mean(train_loss_mean)}, {train_loss_mean}")

        # Validation
        #############

        val_loss_history = []
        inn.eval()

        with torch.no_grad():
            for priors, positions in val_loader:
                noise_batch = torch.randn(batch_size, Z_DIM)
                positions = torch.cat((noise_batch, positions), dim=1)

                # TODO use jac=False ?
                positions_pred, _ = inn(priors)

                batch_losses = []
                batch_losses.extend(loss_forward_mmd(positions_pred, positions))
                batch_losses.append(loss_backward_mmd(priors, positions, inn))
                batch_losses.append(loss_reconstruction(positions_pred.data, positions, priors, inn))

                val_loss_history.append([batch_loss.item() for batch_loss in batch_losses])

            val_loss_mean = np.mean(val_loss_history, axis=0)
            print(f"[Epoch {epoch}] Val loss:   {np.mean(val_loss_mean)}, {val_loss_mean}")


if __name__ == "__main__":
    train()
