# Open source 
This repository is built on the basis of NCNN (https://github.com/atanmarko/ncnn-with-cuda). It supports to distribute most commonly used CNN network over multiple devices/nodes at the edge.
# ncnn (https://github.com/Tencent/ncnn)
ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design. ncnn does not have third party dependencies. it is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, create intelligent APPs, and bring the artificial intelligence to your fingertips. ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

# How to build.

    sudo apt install libopenmpi-dev libopenmpi2:arm64  openmpi-bin openmpi-common
    cmake -DNCNN_VULKAN=OFF -DNCNN_CUDA=ON -DLOG_LAYERS=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_EXAMPLES=ON ..
    cmake -DNCNN_VULKAN=OFF -DNCNN_CUDA=ON -DLOG_LAYERS=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DNCNN_OPENMP=OFF ..
    cmake -DNCNN_VULKAN=OFF -DNCNN_CUDA=OFF -DNCNN_MPI=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/tx2.toolchain.cmake -DNCNN_BUILD_EXAMPLES=ON ..

---
### License

[BSD 3 Clause](LICENSE.txt)

