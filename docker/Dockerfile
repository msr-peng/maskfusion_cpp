FROM bluebirdpp/cudagl:9.0-cudnn7-devel-ubuntu16.04
ARG OPENCV_VERSION=3.4.1
ARG PYTORCH_COMMIT_ID=832216
# build opencv from source
RUN apt-get update &&\
    apt-get install build-essential &&\
    apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev &&\
    cd /home && mkdir src && cd src &&\
    git clone https://github.com/opencv/opencv.git &&\
    cd opencv && git checkout $OPENCV_VERSION && mkdir build && cd build &&\
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. &&\
    make -j`nproc` &&\
    make install

#Build pytorch from source
RUN cd /home/src &&\
    apt-get install python-pip libomp-dev &&\
    git clone https://github.com/pytorch/pytorch.git &&\
    cd pytorch && git checkout $PYTORCH_COMMIT_ID && \
    git submodule update --init --recursive &&\
    pip install -r requirements.txt &&\
    mkdir build && cd build &&\
    cmake -D USE_CUDA=1 -D USE_CUDNN=1 -D USE_OPENCV=1 -D USE_OPENMP=1 -D BUILD_TORCH=1 -D CMAKE_INSTALL_PREFIX=/usr/local .. &&\
    make -j`nproc` &&\
    make install

#Install dependencies of SemanticFusion
RUN cd /home/src &&\
    apt-get install -y cmake-qt-gui libusb-1.0-0-dev libudev-dev openjdk-8-jdk freeglut3-dev libglew-dev libsuitesparse-dev libeigen3-dev zlib1g-dev libjpeg-dev &&\
    git clone https://github.com/stevenlovegrove/Pangolin.git &&\
    cd Pangolin && mkdir build && cd build &&\
    cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON &&\
    make -j8 && cd ../.. &&\
    git clone https://github.com/occipital/OpenNI2.git &&\
    cd OpenNI2 && make -j8 && cd .. &&\
    apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler &&\
    apt-get install --no-install-recommends libboost-all-dev &&\
    apt-get install libopenblas-dev &&\ libgflags-dev libgoogle-glog-dev liblmdb-dev &&\

#Build MaskFusion_cpp
RUN cd /home/src &&\
    git clone https://github.com/msr-peng/maskfusion_cpp.git &&\
    cd maskfusion_cpp && mkdir build && cd build && cmake .. && make -j`nproc` &&\

#Delete downloaded repositories of OpenCV and Pytorch
RUN cd /home/src &&\
    rm -r opencv && rm -r pytorch
