# hash:sha256:3d4782ddeee31fe9b083369ab99d99e1a917efa528cad89a8f20ebd3b0fccf22
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-python3.7.3-cuda10.1-cudnn7-ubuntu18.04-newkey

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install \
    --upgrade \
    pip

RUN conda install -y \
        cmake \
        scikit-image 

RUN pip install -U --no-cache-dir \
    facenet-pytorch \
    gdown \
    imageio \
    matplotlib \
    mlxtend==0.20.0 \
    dlib \
    opencv-python \
    seaborn \
    torch==1.13 \
    torchvision

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y