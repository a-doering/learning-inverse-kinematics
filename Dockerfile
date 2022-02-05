FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV PYTHONPATH="/learning-inverse-kinematics/src"

RUN apt update \
    && apt install -y htop python3-dev wget git imagemagick

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda create -y -n .venv python=3.7

 # Installation of packages needed for rokin for 3d kinematics
ENV EIGEN_INCLUDE_DIR="/usr/include/eigen3/"
RUN apt install -y libeigen3-dev\
    && apt install -y swig \
    && apt install -y gfortran
# Needed for rokin to not throw an error, modify if you want to use meshes
ENV ROKIN_MESH_DIR="your/path/to/the/meshes/"

COPY . learning-inverse-kinematics/
WORKDIR /learning-inverse-kinematics

RUN /bin/bash -c "source activate .venv \
    && pip install -r requirements.txt \
    && pip install git+https://github.com/scleronomic/rokin@stable1.0 \
    && pip install git+https://github.com/VLL-HD/FrEIA@v0.2"

# Start conda environment when a container is started
CMD ["/bin/bash", "source activate .venv"]
RUN echo "source activate .venv" >> ~/.bashrc