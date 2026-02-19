## will make sure code is anonymous before uploading, before Feb 23;

## Docker is available

A Dockerfile is available in the folder container. (make sure you have space available which is larger than 32G)

change start_interative.sh accordingly to mount your space
pretrained model and test data to boomer2_tools folder


```
cd container
./build.sh  # It takes at least 20-30 mins
./start_interative.sh
```
in the docker environment, an additional step is needed for torchvision (It's the same step as in the end of DockerFile, without the following extra step, torchvision cannot run properly with cuda, see FAQ)
```
cd /vision/build
rm -rf *
# Add -DWITH_CUDA=on support for the CUDA if needed
cmake -DCMAKE_PREFIX_PATH=/libtorch -DWITH_CUDA=on .. 
make -j8
make install
```

```
cd /your-boomer-tool-path
mkdir build
cd build
cp CMakeLists.txt CMakeLists.txt.bak
mv CMakeLists_docker.txt CMakeLists.txt
cmake ..   # cmake -DUSE_CUDA=OFF   if you don't have CUDA
make -j8
```
An test example:
```
./testLibTorch --cpu --model ../models/my_module_model_new.pt --input_depth ../data/test/DepthImages/155_depth_0.png --input_intensity ../data/test/IntensityImages/155_intensity_0.png 
```
To test libtorch with cuda version, remove --cpu or change --cpu to --gpu


Match example:
```
./matcher --cpu --input ../data/pcd-153.pcd --model ../models/my_module_model_new.pt --feature PFH  --cam_z -235 --cam_x 1 --cam_y 1 --ransac_n_closest             

```
If torchvision is build without -DWITH_CUDA=ON, matcher has to run with argument --cpu if you want to use learned detector. Otherwise, the code will decide to run with cuda or not itself. If cuda is available on your machine but torchvision is compile without cuda, then it will be a problem.


```
./matcher  --input ../data/simulation_scan/with_intensity/28_-2_-12.pcd --input_2 ../data/simulation_scan/with_intensity/27_-2_-18.pcd --model ../models/new_bolts/custom_faster_rcnn_chkpt_49.pt --fisheye --feature PFH  --cam_z 0 --cam_x 0 --cam_y 0 --coffset_x 0 --coffset_y 0 --coffset_z 0 --cam_x_gt -1 --cam_y_gt 6 --cam_z_gt 0 --ransac_n_closest --ransac_forward_reverse --connect_pts
```
This command will try to match two point clouds, stitch them, and save them. In the stitched point cloud, they are in two different colors.


## Use libtorch
https://pytorch.org/cppdocs/installing.html
```
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+2Bcu118.zip
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

if you don't have cuda or gpu on your machine, use this link:
https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.1%2Bcpu.zip

## Build torchvision
https://github.com/pytorch/vision#using-the-models-on-c
Download torchvision
```
git clone https://github.com/pytorch/vision.git
cd vision
mkdir build
cd build
\# Add -DWITH_CUDA=on support for the CUDA if needed
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DWITH_CUDA=on .. \# path to libtorch need to be an absolute path
make
make install
```

## Build TEASER
https://github.com/MIT-SPARK/TEASER-plusplus.git

```
git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus && mkdir build && cd build
cmake .. && make
sudo make install
sudo ldconfig
cd .. && cd examples/teaser_cpp_ply && mkdir build && cd build
cmake .. && make
./teaser_cpp_ply
```
In CMakeList.txt
```
find_package(teaserpp REQUIRED)

target_link_libraries(xxx 
...
teaserpp::teaser_registration 
teaserpp::teaser_features
...)


```
## Build excutables
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch"  ..
(or cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. # might not work )
make -j8
```
## pretrained model and test data
find the test data and model here: 

## Test
```
./testLibTorch ../models/my_module_model_new.pt ../data/test/DepthImages/155_depth_0.png ../data/test/IntensityImages/155_intensity_0.png 
```


##  FAQ
- In docker version, you might get error when runing the libTorchTest 
  ```
  terminate called after throwing an instance of 'std::runtime_error'
  what():  The following operation failed in the TorchScript interpreter.
  ...
  return torch.ops.torchvision.nms(boxes, scores, iou_threshold)
           ~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
  RuntimeError: CUDA error: no kernel image is available for execution on the device
  CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
  For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
  Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

  ```
  You might forgot to build torchvision again. 
  ```
  cd /vision/build
  rm -rf *
  # Add -DWITH_CUDA=on support for the CUDA if needed
  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DWITH_CUDA=on .. # path to libtorch need to be an absolute path
  make -j8
  make install
  ```
- Target "xxx" requires the language dialect "CUDA17" (with compiler extensions), but CMake does not know the compile flags to use to enable it.
  - It might be the problem of cmake version. It needs cmake version > 3.18
  - CUDA_STANDARD 17 and above is only supported on CMAKE 18 and above. https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_STANDARD.html
- link problem with opencv and pcl.
  - ```
    undefined reference to `cv::imread(std::string const&, int)' 
    ```
  - https://medium.com/mlearning-ai/pytorch-c-4-using-torchvision-models-f12bd25f4744
  - use cxx11-ABI instead of Pre-cxx11 ABI.
  - use-D GLIBCXX_USE_CXX11_ABI=0 when building openCV and PCL (not recommend)

- cmake find different version of cuda
  - ```
    CMake Error at /libtorch/share/cmake/Caffe2/public/cuda.cmake:65 (message):
    Found two conflicting CUDA installs:
    V11.8.89 in '/usr/local/cuda-11.8/include' and  
    V11.8.89 in '/usr/local/cuda/include'
    ```

- check settings in CMakeLists.txt  
    ```
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    ```
  - check the environment variables $CUDA_PATH and $CUDA_HOME have the same value as CMAKE_CUDA_COMPILER, otherwise 
    ```
    export CUDA_PATH=/path_to_your_cuda/cuda
    export CUDA_HOME=/path_to_your_cuda/cuda
    ```
    e.g.:
    ```
    export CUDA_HOME=/usr/local/cuda-11.8
    export CUDA_HOME=/usr/local/cuda-11.8
    ```
- Be careful about the scope of tensor variables.
  

