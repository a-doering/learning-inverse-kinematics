from FrEIA.framework import SequenceINN
from FrEIA.modules import RNVPCouplingBlock
from torch import nn


def create_inn(n_dim: int = 2) -> SequenceINN:
    # simple chain of RealNVP coupling blocks
    inn = SequenceINN(n_dim)
    for k in range(3):
        inn.append(RNVPCouplingBlock, subnet_constructor=_subnet_fc)

    return inn


def _subnet_fc(dims_in, dims_out) -> nn.Sequential:
    """Creates a subnet for use inside an affine coupling block"""
    return nn.Sequential(
        nn.Linear(dims_in, 256),
        nn.ReLU(),
        nn.Linear(256, dims_out),
    )
