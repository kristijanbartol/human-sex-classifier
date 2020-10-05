FROM ubuntu:18.04

WORKDIR /gender-classifier/

RUN apt-get update \
    && apt-get install -y \
        python3-pip \
        vim \
        unzip \
	wget \
	git \
	libsm6 \
	libxext6 \
	libxrender-dev \
	parallel \
	imagemagick
RUN pip3 install numpy \
#    torch \
#    torchvision \
    matplotlib \
    scikit-image \
    pandas \
    progress \
    h5py
#    opencv-python

RUN pip3 install --pre torch torchvision \
    -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
