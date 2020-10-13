FROM ubuntu:18.04

WORKDIR /gender-classifier/

RUN apt-get update \
    && apt-get install -y \
	cmake \
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
RUN pip3 install scikit-build
RUN pip3 install numpy \
#    torch \
#    torchvision \
    matplotlib \
    scikit-learn \
    scikit-image \
    pandas \
    progress \
    h5py \
    opencv-python \
    tensorflow \
    tensorboard>=1.15

RUN pip3 install --pre torch torchvision \
    -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
