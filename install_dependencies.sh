#!/bin/sh
install_cmake() {
    echo "instaling curl & download cmake"
    CMAKETAR=cmake-3.20.4.tar.gz
    sudo apt install -y curl libssl-dev
    curl -OL https://github.com/Kitware/CMake/releases/download/v3.20.4/cmake-3.20.4-SHA-256.txt
    [ -f ~/cmake-3.20.4.tar.gz ] && echo "$CMAKETAR exist." || curl -OL https://github.com/Kitware/CMake/releases/download/v3.20.4/cmake-3.20.4.tar.gz
    sha256sum -c --ignore-missing cmake-3.20.4-SHA-256.txt
    tar xvzf cmake-3.20.4.tar.gz && cd cmake-3.20.4 && ./configure --prefix=/usr/local && make -j6 && sudo make install
}

install_ucx(){
    echo "installing ucx first."
    echo "---------------------"
    sudo apt install hwloc
    git clone https://github.com/openucx/ucx.git ucx
    cd ucx && ./autogen.sh 
    mkdir -p build 
    cd build && sudo ../contrib/configure-release --prefix=/usr/local --with-cuda=/usr/local/cuda
    sudo make -j6 && sudo make install
    cd ~
}
install_mpich4(){
    cd ~
    echo "installing mpich 4.0.1"
    sudo apt install -y perl firewalld
    [ -f ~/openmpi-4.1.0.tar.gz ] && echo "$CMAKETAR exist." || curl -OL https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.0.tar.gz
    tar xvzf openmpi-4.1.0.tar.gz && cd openmpi-4.1.0 
    ./configure --prefix=/usr/local --with-cuda --with-ucx  --with-hwloc --enable-heterogeneous --with-slurm=no
    make -j6 && sudo make install
}
install_sen(){
    echo "installing SEN library for CNN inference."
    echo "-----------------------------------------"
    cd ~/SEN/
    sudo apt install -y build-essential git libomp-dev libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils
    mkdir -p build && cd build
    cmake -DNCNN_VULKAN=ON -DNCNN_CUDA=OFF -DNCNN_MPI=ON -DNCNN_OPENMP=ON -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=ON -DNCNN_BENCHMARK=OFF -DCMAKE_TOOLCHAIN_FILE=../toolchains/tx2.toolchain.cmake ..
    make -j6 
}
install_cmake
install_ucx
install_mpich4
install_sen
