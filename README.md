# Advanced Deep Learning for Robotics Project

## Kinematics
2D: Forward and inverse kinematics with visualization for a robot arm with at least one prismatic and multiple rotational joints. The inverse is implemented with rejection sampling.



Forward PRRR           |  Inverse PRRR
:-------------------------:|:-------------------------:
![](img/forward.png)  |  ![](img/inverse.png)

## GAN
### Conditional GAN

Training without discriminator, 7DOF, mode collapse can be seen.


Multiple Positions, no discriminator:
![](img/evaluate_no_disc_multiple_pos.png)
Latent variable walk, no discriminator:
![](img/evaluate_no_disc_latent_walk.png) We can observe that z has little influence, altough z is totally out of the distribution.
### Info Gan
![](img/infogan_2d/evaluate_latent_walk_z_2_0.01.png)
![](img/infogan_2d/evaluate_latent_walk_z_5_0.png)
![](img/infogan_2d/evaluate_latent_walk.png)
![](img/infogan_2d/evaluate_multiple_pos.png)

![](img/infogan_2d/infogan_2d.gif)

## Setup

```sh
# Git setup
mkdir code
cd code
git clone https://github.com/a-doering/tum-adlr-ss21-01.git
cd tum-adlr-ss21-01
git checkout #--track origin/my_branch 

# Virtual Env Setup
python3 -m pip install virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
# Currently there was an import error for freia, is not installed yet on remote, can remove with nano from requirements.txt for now

# Installation of packages needed for 3d kinematics
sudo apt install libeigen3-dev
#  export EIGEN_INCLUDE_DIR="your/path/to/eigen3/"
export EIGEN_INCLUDE_DIR="/usr/include/eigen3/" # on my vm
sudo apt install swig
sudo apt install gfortran
pip install git+https://github.com/scleronomic/rokin

# For 3D visualization, one can use meshes, export the path
export ROKIN_MESH_DIR="your/path/to/the/meshes/"

# Generate training data
python src/kinematics/robot_arm_2d_torch.py
# Train
python src/main.py
# Now login with wandb when asked

# Errors
# In case of error you can run "wandb off" to switch the syncing off
```
### Create gif
Install: `sudo apt install imagemagick`, run in the directory with images:
```sh
convert -delay 100 -loop 0 $(ls -1v plot*) training.gif
```