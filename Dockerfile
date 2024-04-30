# Start from NVIDIA CUDA container (Ubuntu 20.04)
FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set as non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Copy necessary files
COPY requirements.txt /tmp/
COPY tls2trees /opt/tls2trees

# Update, install packages, and clean up
RUN apt-get update && \
    apt-get install -y apt-utils git curl vim unzip wget build-essential python3-pip cuda-compat-11-3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip requirements and clean up
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Symlink for libcusolver and make scripts executable
RUN ln -s /usr/local/cuda-11.3/lib64/libcusolver.so.11 /usr/local/cuda-11.3/lib64/libcusolver.so.10 && \
    chmod a+x /opt/tls2trees/semantic.py /opt/tls2trees/instance.py

# Set environmental variables
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib:/usr/local/cuda-11.3/lib64:/usr/local/cuda/compat
ENV PYTHONPATH=/opt/:$PYTHONPATH
ENV PATH=/opt/tls2trees:$PATH
