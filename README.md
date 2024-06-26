# LiKL-Cpp
Cpp implementation of "LiKL: A Lightweight Approach for Joint Detection and Description of Keypoint and Line".
Which is capable of simultaneously extracting keypoint and line features in an image and inferring feature descriptors in a single process.

The pytorch implementation in [LiKL](https://github.com/hjshxb/LiKL/)

## Updates
### 2024-04-28
- Add TensorRT support. (Only support static input shapes now.)

## Usage
### 1. Dependency
- Eigen3
```shell
sudo apt install libeigen3-dev
```
- google-glog
```shell
sudo apt install libgoogle-glog-dev
```
- OpenCV4
- libtorch
```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip
unzip libtorch
```
- torchvision
```shell
wget https://github.com/pytorch/vision/archive/refs/tags/v0.12.0.zip
unzip vision-0.12.0.zip

## Build torchvision
cd vision
mkdir build && cd build
cmake .. -DWITH_CUDA=on -DUSE_PYTHON=on -DCMAKE_PREFIX_PATH=/path/to/libtorch
make
make install
```

- TensorRT (Optional)
Tested in 8.6.1.6

### 2. Build
```shell
mkdir build && cd build
# Add -DWITH_TENSORRT=on support for the TensorRT if needed
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make
```

### 3. Download the converted model
#### TorchScript model
You can download from [Baidu (y3yu) ](https://pan.baidu.com/s/1gIeHr6EWJF-80i0sGX-lYQ) or [Onedrive](https://1drv.ms/u/s!Ah0c5cK_vtly10KVn5ypGBM_sBRS?e=frMbm8).

You can also convert pytorch model to torchcript. Please refer to [LiKL](https://github.com/hjshxb/LiKL/).

#### ONNX model
You can convert pytorch model to onnx. Please refer to [LiKL](https://github.com/hjshxb/LiKL/).


### 4. demo
Set the torchscript path in `configs/likl.yaml`
```shell
./build/test_likl ./asset/terrace0.JPG ./asset/terrace1.JPG
```

### 5. Acknowledgements
Thanks to [SuperPoint-SuperGlue-TensorRT)](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT)




