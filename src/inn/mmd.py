import numpy as np
import torch

# This file implements the maximum mean discrepancy (MMD), a distance-measure between probability distributions.
# MMD was introduced in this paper: Gretton et al., "A Kernel Two-Sample Test", 2012.
#                                   https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf

# Slides by one of the authors explaining MMD: http://alex.smola.org/talks/taiwan_4.pdf (the first section is relevant)

# For a more intuitive explanation, see this answer on stats.stackexchange.com:
# https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution

# As a kernel, we use the inverse multiquadratics kernel  k(x, y) = C / (C + ||x - y||_2^2)  for vectors x, y
# Introduced in this paper: Tolstikhin et al., "Wasserstein Auto-Encoders", 2017. https://arxiv.org/pdf/1711.01558.pdf
# We use C = 1.

# This code is inspired by
# https://github.com/VLL-HD/analyzing_inverse_problems/blob/6f1334622995/inverse_problems_science/losses.py


MMD_FORWARD_KERNELS = [(0.2, 2), (1.5, 2), (3.0, 2)]
MMD_BACKWARD_KERNELS = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]


def _mmd_matrix_multiscale(x, y, widths_exponents):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX = torch.zeros(xx.shape)
    YY = torch.zeros(xx.shape)
    XY = torch.zeros(xx.shape)

    for C, a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return XX + YY - 2.*XY


def forward_mmd(y0, y1):
    return _mmd_matrix_multiscale(y0, y1, MMD_FORWARD_KERNELS)


def backward_mmd(x0, x1):
    return _mmd_matrix_multiscale(x0, x1, MMD_BACKWARD_KERNELS)


def l2_fit(input, target, batch_size: int = 50):
    return torch.sum((input - target)**2) / batch_size


def l2_dist_matrix(x, y):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    return torch.clamp(rx.t() + ry - 2. * xy, 0, np.inf)
