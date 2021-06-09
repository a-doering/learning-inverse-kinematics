# Advanced Deep Learning for Robotics Project

## Kinematics
2D: Forward and inverse kinematics with visualization for a 4 DOF PRRR arm. Inverse is implemented with gradient descent, (not optimized) runtime: 99.5s for 100 forward passes with 1000 inverse passes each.

Forward             |  Inverse
:-------------------------:|:-------------------------:
![](img/forward.png)  |  ![](img/inverse.png)


## Setup

```sh
# Git setup
mkdir code
cd code
git clone https://github.com/a-doering/tum-adlr-ss21-01.git
cd tum-adlr-ss21-01
git checkout --track origin/feature/gan 

# Virtual Env Setup
python3 -m pip install virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
# Currently there was an import error for freia, is not installed yet on remote, can remove with nano from requirements.txt for now

# Generate training data
python src/kinematics/robot_arm_2d.py
# Train
python src/main.py
# Now login with wandb when asked

# Errors
# In case of error you can run "wandb off" to switch the syncing off
```

### Remote Setup
TODO