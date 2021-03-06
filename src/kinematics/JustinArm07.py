import numpy as np
import torch
from torch import nn

pi = np.pi


class JustinArm07Net(nn.Module):
    def __init__(self):
        super(JustinArm07Net, self).__init__()
        self.n_frames = 8
    
    def forward(self, x):
        q = x
        f = torch.zeros(tuple(q.shape)[:-1] + (self.n_frames, 4, 4), device=x.device)
        
        # Fill frames
        f[..., 0, 0, 0] = torch.cos(q[..., 0])
        f[..., 0, 0, 1] = -torch.sin(q[..., 0])
        f[..., 0, 0, 2] = 0
        f[..., 0, 0, 3] = 0
        f[..., 0, 1, 0] = torch.sin(q[..., 0])
        f[..., 0, 1, 1] = torch.cos(q[..., 0])
        f[..., 0, 1, 2] = 0
        f[..., 0, 1, 3] = 0
        f[..., 0, 2, 0] = 0
        f[..., 0, 2, 1] = 0
        f[..., 0, 2, 2] = 1
        f[..., 0, 2, 3] = 0
        f[..., 0, 3, 0] = 0
        f[..., 0, 3, 1] = 0
        f[..., 0, 3, 2] = 0
        f[..., 0, 3, 3] = 1
        
        f[..., 1, 0, 0] = torch.cos(q[..., 1])
        f[..., 1, 0, 1] = -torch.sin(q[..., 1])
        f[..., 1, 0, 2] = 0
        f[..., 1, 0, 3] = 0
        f[..., 1, 1, 0] = 0
        f[..., 1, 1, 1] = 0
        f[..., 1, 1, 2] = -1
        f[..., 1, 1, 3] = 0
        f[..., 1, 2, 0] = torch.sin(q[..., 1])
        f[..., 1, 2, 1] = torch.cos(q[..., 1])
        f[..., 1, 2, 2] = 0
        f[..., 1, 2, 3] = 0
        f[..., 1, 3, 0] = 0
        f[..., 1, 3, 1] = 0
        f[..., 1, 3, 2] = 0
        f[..., 1, 3, 3] = 1
        
        f[..., 2, 0, 0] = torch.cos(q[..., 2] - 0.5*pi)
        f[..., 2, 0, 1] = -torch.sin(q[..., 2] - 0.5*pi)
        f[..., 2, 0, 2] = 0
        f[..., 2, 0, 3] = 0
        f[..., 2, 1, 0] = 0
        f[..., 2, 1, 1] = 0
        f[..., 2, 1, 2] = 1
        f[..., 2, 1, 3] = 0.4
        f[..., 2, 2, 0] = -torch.sin(q[..., 2] - 0.5*pi)
        f[..., 2, 2, 1] = -torch.cos(q[..., 2] - 0.5*pi)
        f[..., 2, 2, 2] = 0
        f[..., 2, 2, 3] = 0
        f[..., 2, 3, 0] = 0
        f[..., 2, 3, 1] = 0
        f[..., 2, 3, 2] = 0
        f[..., 2, 3, 3] = 1
        
        f[..., 3, 0, 0] = torch.cos(q[..., 3])
        f[..., 3, 0, 1] = -torch.sin(q[..., 3])
        f[..., 3, 0, 2] = 0
        f[..., 3, 0, 3] = 0
        f[..., 3, 1, 0] = 0
        f[..., 3, 1, 1] = 0
        f[..., 3, 1, 2] = -1
        f[..., 3, 1, 3] = 0
        f[..., 3, 2, 0] = torch.sin(q[..., 3])
        f[..., 3, 2, 1] = torch.cos(q[..., 3])
        f[..., 3, 2, 2] = 0
        f[..., 3, 2, 3] = 0
        f[..., 3, 3, 0] = 0
        f[..., 3, 3, 1] = 0
        f[..., 3, 3, 2] = 0
        f[..., 3, 3, 3] = 1
        
        f[..., 4, 0, 0] = -torch.cos(q[..., 4])
        f[..., 4, 0, 1] = torch.sin(q[..., 4])
        f[..., 4, 0, 2] = 0
        f[..., 4, 0, 3] = 0
        f[..., 4, 1, 0] = 0
        f[..., 4, 1, 1] = 0
        f[..., 4, 1, 2] = 1
        f[..., 4, 1, 3] = 0.390
        f[..., 4, 2, 0] = torch.sin(q[..., 4])
        f[..., 4, 2, 1] = torch.cos(q[..., 4])
        f[..., 4, 2, 2] = 0
        f[..., 4, 2, 3] = 0
        f[..., 4, 3, 0] = 0
        f[..., 4, 3, 1] = 0
        f[..., 4, 3, 2] = 0
        f[..., 4, 3, 3] = 1
        
        f[..., 5, 0, 0] = torch.cos(q[..., 5] + 0.5*pi)
        f[..., 5, 0, 1] = -torch.sin(q[..., 5] + 0.5*pi)
        f[..., 5, 0, 2] = 0
        f[..., 5, 0, 3] = 0
        f[..., 5, 1, 0] = 0
        f[..., 5, 1, 1] = 0
        f[..., 5, 1, 2] = -1
        f[..., 5, 1, 3] = 0
        f[..., 5, 2, 0] = torch.sin(q[..., 5] + 0.5*pi)
        f[..., 5, 2, 1] = torch.cos(q[..., 5] + 0.5*pi)
        f[..., 5, 2, 2] = 0
        f[..., 5, 2, 3] = 0
        f[..., 5, 3, 0] = 0
        f[..., 5, 3, 1] = 0
        f[..., 5, 3, 2] = 0
        f[..., 5, 3, 3] = 1
        
        f[..., 6, 0, 0] = torch.cos(q[..., 6] - 0.5*pi)
        f[..., 6, 0, 1] = -torch.sin(q[..., 6] - 0.5*pi)
        f[..., 6, 0, 2] = 0
        f[..., 6, 0, 3] = 0
        f[..., 6, 1, 0] = 0
        f[..., 6, 1, 1] = 0
        f[..., 6, 1, 2] = -1
        f[..., 6, 1, 3] = 0
        f[..., 6, 2, 0] = torch.sin(q[..., 6] - 0.5*pi)
        f[..., 6, 2, 1] = torch.cos(q[..., 6] - 0.5*pi)
        f[..., 6, 2, 2] = 0
        f[..., 6, 2, 3] = 0
        f[..., 6, 3, 0] = 0
        f[..., 6, 3, 1] = 0
        f[..., 6, 3, 2] = 0
        f[..., 6, 3, 3] = 1
        
        f[..., 7, 0, 0] = -1.
        f[..., 7, 0, 1] = 0
        f[..., 7, 0, 2] = 0
        f[..., 7, 0, 3] = 0
        f[..., 7, 1, 0] = 0
        f[..., 7, 1, 1] = 0
        f[..., 7, 1, 2] = 1.
        f[..., 7, 1, 3] = 0.118
        f[..., 7, 2, 0] = 0
        f[..., 7, 2, 1] = 1.
        f[..., 7, 2, 2] = 0
        f[..., 7, 2, 3] = 0
        f[..., 7, 3, 0] = 0
        f[..., 7, 3, 1] = 0
        f[..., 7, 3, 2] = 0
        f[..., 7, 3, 3] = 1.

        # Combine frames
        f[..., 1, :, :] = f[..., 0, :, :] @ f[..., 1, :, :]
        f[..., 2, :, :] = f[..., 1, :, :] @ f[..., 2, :, :]
        f[..., 3, :, :] = f[..., 2, :, :] @ f[..., 3, :, :]
        f[..., 4, :, :] = f[..., 3, :, :] @ f[..., 4, :, :]
        f[..., 5, :, :] = f[..., 4, :, :] @ f[..., 5, :, :]
        f[..., 6, :, :] = f[..., 5, :, :] @ f[..., 6, :, :]
        f[..., 7, :, :] = f[..., 6, :, :] @ f[..., 7, :, :]
        
        return f


def test():
    from rokin.Robots import JustinArm07

    robot = JustinArm07()
    net = JustinArm07Net()
    q = robot.sample_q((1000, 3))

    f = robot.get_frames(q)

    q_t = torch.tensor(q)
    f_t = net.forward(q_t)

    print(np.allclose(f, f_t, atol=1e-6))

    d = f - np.array(f_t)
    print(np.abs(d).max())

if __name__ == "__main__":
    test()