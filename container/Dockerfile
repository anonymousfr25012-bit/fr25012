FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="xxx"

SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Stockholm

RUN apt update -y && apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev  libgl1-mesa-glx cmake git vim gedit htop

#Make sure python is installed
RUN apt-get install -y --no-install-recommends python3-pip python3 vim curl gnupg wget bash-completion xterm less gdb libeigen3-dev libpcl-dev zip unzip

# Dependencies
RUN apt-get install -y build-essential cmake libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran -y

RUN git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git \
    && cd opencv_contrib && git checkout 4.8.1 \
    && cd ../opencv && mkdir -p build \
    && git checkout 4.8.1 

RUN cd opencv/build && cmake -D CMAKE_BUILD_TYPE=Release -D  CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules .. \
  && make -j8 && make install && rm -rf /opencv /opencv_contrib
  
 
#install useful python packages
RUN pip3 install pandas matplotlib scipy scikit-learn scikit-image imgaug opencv-python IPython[all] matplotlib Cython requests PyYAML h5py tqdm Pillow tensorboard

# install pytorch
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118


# libtorch
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip 
#RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.1.0%2Bcu118.zip

RUN unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip && rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip -f 
#RUN unzip libtorch-shared-with-deps-2.1.0+cu118.zip && rm libtorch-shared-with-deps-2.1.0+cu118.zip  

# torch-vision
RUN apt update
RUN git clone https://github.com/pytorch/vision.git
# RUN cd vision && git checkout v0.16.0 &&mkdir build && cd build 
RUN cd vision && git checkout v0.16.0 && mkdir build && cd build \
    && cmake -DWITH_CUDA=on -DCMAKE_PREFIX_PATH=/libtorch  .. \
    && make -j8 && make install 

# TEASER++
RUN apt-get -y install python3-pybind11
RUN git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
RUN cd TEASER-plusplus && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TEASER_FPFH=ON  -DBUILD_PYTHON_BINDINGS=ON .. && make && make install

# clean
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

RUN ldconfig


