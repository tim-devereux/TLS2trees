# Use the official PyTorch image with CUDA 11.1
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    liblaszip-dev \
    git \
    curl \
    wget \
    sudo \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt /tmp/requirements.txt

# Install pip packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
    --find-links https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html -r /tmp/requirements.txt

# Install torch-geometric dependencies
RUN pip install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-geometric

# Install laszip
RUN git clone https://github.com/LASzip/LASzip.git /tmp/LASzip \
    && cd /tmp/LASzip \
    && mkdir build && cd build \
    && cmake .. \
    && make \
    && make install \
    && ldconfig \
    && rm -rf /tmp/LASzip

# Create a non-root user 'vscode' with sudo privileges
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up working directory
WORKDIR /workspaces

# Switch to 'vscode' user
USER $USERNAME

# Default command
CMD ["bash"]
