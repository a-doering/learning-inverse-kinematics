#!/bin/bash
git clone https://github.com/a-doering/learning-inverse-kinematics.git
cd learning-inverse-kinematics
docker build -f Dockerfile -t learnik .
# This will also activate the conda environment
docker run -ti learnik /bin/bash