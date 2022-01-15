#!/bin/bash
git clone https://github.com/a-doering/tum-adlr-ss21-01.git
cd tum-adlr-ss21-01
docker build -f Dockerfile -t adlr .
# This will also activate the conda environment
docker run -ti adlr /bin/bash