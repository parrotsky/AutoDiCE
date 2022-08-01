# README

# Open source

This repository ![AutoDiCE](20220801.png)


The inference library is built on the basis of [NCNN](https://github.com/atanmarko/ncnn-with-cuda). And it extends with MPI Interface to support Multi-node Inference. This library is used to distribute the most commonly used CNN network over multiple devices/nodes at the edge. 

## ncnn

ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design. ncnn does not have third-party dependencies. it is cross-platform, and runs faster than all known open source frameworks on mobile phone CPU. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, creating intelligent APPs, and bringing artificial intelligence to your fingertips. ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu, and so on.

# How to build.

## prerequisites

### Installing MPI

MPI is simply a standard interface for others to follow in their implementation. Because of this, there are a wide variety of MPI implementations out there, such as [OpenMPI](https://www.open-mpi.org/software/ompi/v4.1/), [MPICH]([https://www.mpich.org/](https://www.mpich.org/)). MPI is used for multi-node communication due to its outstanding performance. Users are free to use any implementation they wish, but only the limitation for installing MPI is to ensure that the MPI version keeps consistent on every device.

### Installing Dependencies & AutoDiCE

We also provide an [installation](./install_dependencies.sh) script.

```
# install cmake --version 3.20
# install Opencv & protobuf & vulkan (if needed)
sudo apt install build-essential git libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
cd AutoDiCE && mkdir -p build && cd build
### Laptop with GPUs…
cmake -DNCNN_VULKAN=OFF -DNCNN_CUDA=ON -DLOG_LAYERS=ON -DNCNN_MPI=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_EXAMPLES=ON ..

###  NVIDIA Jetson NANO/TX2/NX series
cmake -DNCNN_VULKAN=ON -DNCNN_CUDA=OFF -DLOG_LAYERS=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DNCNN_OPENMP=OFF ..

###  CPU-only Machine
cmake -DNCNN_VULKAN=OFF -DNCNN_CUDA=OFF -DNCNN_MPI=ON -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_EXAMPLES=ON ..
```

# how to use AutoDiCE

please check our step-by-step tutorial.[Tutorial](tutorial.md)

### License

[BSD 3 Clause](LICENSE.txt)
