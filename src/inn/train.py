import csv
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import load_dataset
from model import create_inn
import mmd


# factors for the loss terms
FORWARD_FIT_FACTOR = 1
FORWARD_MMD_FACTOR = 50
BACKWARD_MMD_FACTOR = 500
RECONSTRUCTION_FACTOR = 1


def run_epoch(
    inn: nn.Module,
    data_loader: DataLoader,
    optimizer: Optional[Adam],
    batch_size: int,
    position_dim: int,
    z_dim: int,
) -> Tuple[float, Tuple[float, float, float, float]]:
    """
    `optimizer` should be `None` during validation.
    :return: 1. The average loss in this epoch
             2. The individual, weighted loss terms (forward L2 fit, forward MMD, backward MMD, reconstruction L2 fit)
    """
    loss_history = []

    for priors, positions in data_loader:
        noise_batch = torch.randn(batch_size, z_dim)
        positions = torch.cat((noise_batch, positions), dim=1)

        # TODO use jac=False for inn(...)?

        # forward loss
        positions_pred, _ = inn(priors)
        # remove gradients wrt positions
        output_block_grad = torch.cat((positions_pred[:, :z_dim], positions_pred[:, -position_dim:].data), dim=1)
        loss_forward_fit = FORWARD_FIT_FACTOR * mmd.l2_fit(positions_pred[:, z_dim:], positions[:, z_dim:])
        loss_forward_mmd = FORWARD_MMD_FACTOR * torch.mean(mmd.forward_mmd(output_block_grad, positions))

        # backward loss
        priors_pred, _ = inn(positions, rev=True)
        loss_backward_mmd = BACKWARD_MMD_FACTOR * torch.mean(mmd.backward_mmd(priors, priors_pred))

        # reconstruction loss
        x_reconstructed, _ = inn(positions_pred.data, rev=True)
        loss_reconstruction = RECONSTRUCTION_FACTOR * mmd.l2_fit(x_reconstructed, priors)

        batch_losses = [loss_forward_fit, loss_forward_mmd, loss_backward_mmd, loss_reconstruction]
        loss_history.append([batch_loss.item() for batch_loss in batch_losses])

        total_batch_loss = sum(batch_losses)

        if optimizer:
            total_batch_loss.backward()
            optimizer.step()

    loss_mean = np.mean(loss_history, axis=0)
    return sum(loss_mean), tuple(loss_mean)


def train(
    batch_size: int = 128,
    lr: float = 1e-6,
    epochs: int = 10000000,
    lr_scheduler_patience: int = 10,
    val_set_portion: float = 0.1,  # portion of the dataset that will be used for validation
    log_file: str = "losses.csv",
):
    # prepare dataset
    dataset, priors_dim, position_dim = load_dataset()
    z_dim = priors_dim - position_dim
    val_set_size = int(len(dataset) * val_set_portion)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_set_size, val_set_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)

    inn = create_inn(priors_dim)
    optimizer = Adam(inn.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, verbose=True)

    for epoch in range(epochs):

        # Training
        ###########

        inn.train()

        train_loss, train_loss_terms = run_epoch(inn, train_loader, optimizer, batch_size, position_dim, z_dim)

        lr_scheduler.step(train_loss)
        print(f"[Epoch {epoch}] Train loss: {train_loss}, {train_loss_terms}")

        # Validation
        #############

        inn.eval()

        with torch.no_grad():
            val_loss, val_loss_terms = run_epoch(inn, val_loader, None, batch_size, position_dim, z_dim)

        print(f"[Epoch {epoch}] Val loss:   {val_loss}, {val_loss_terms}")

        # log losses
        with open(log_file, 'a+', newline='') as file:
            csv.writer(file).writerow([train_loss, *train_loss_terms, val_loss, *val_loss_terms])


if __name__ == "__main__":
    train()
