# Advanced Deep Learning for Robotics Project

## Kinematics
2D: Forward and inverse kinematics with visualization for a 4 DOF PRRR arm. Inverse is implemented with gradient descent, (not optimized) runtime: 99.5s for 100 forward passes with 1000 inverse passes each.

Forward             |  Inverse
:-------------------------:|:-------------------------:
![](img/forward.png)  |  ![](img/inverse.png)


### Previous:

Base from [github](https://github.com/Kartik17/Robotic_Arm), work in progress. Uses the modified DH (Denvavit Hartenberg) [convention](http://www-scf.usc.edu/~csci545/slides/Lect5_Forward-InverseKinematicsII_Short.pdf) as used in introduction to robotics by Craig. Alternatively the kinematics used in INN paper can be used, code on [their github](https://github.com/VLL-HD/inn_toy_data/blob/master/kinematics.py).