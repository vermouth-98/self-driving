FROM python:3
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND noninteractive


WORKDIR ./local/bin
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install numpy timm
RUN apt-get install python3-libnvinfer-dev --assume-yes
RUN pip install --upgrade pip
RUN pip install tensorboardX mss opencv-python Pillow scipy tqdm scipy scikit-image
RUN pip install easydict pandas seaborn Cython h5py six tb-nightly future yacs gdown flake8 yapf isort==4.3.21 imageio matplotlib requests
RUN pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip
WORKDIR /root/
COPY . .
CMD ["python3", "./main.py"]
