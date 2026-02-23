# Docker Setup

A `Dockerfile` is available in the `container/` folder.

> Make sure you have **at least 32GB of free disk space** before building.

Modify `start_interative.sh` to mount your local workspace correctly, including:
- Your working directory
- Pretrained models
- Test data  
(These should be mounted into the `boomer_localization` folder inside the container.)

---

## Build and Start Docker

```bash
cd container
./build.sh          # Takes approximately 20–30 minutes
./start_interative.sh
```
## Important: Rebuild torchvision (Required Inside Docker)
Inside the Docker environment, you must rebuild torchvision with CUDA support.

Without this step, torchvision CUDA operations (e.g., NMS) may fail.
```bash
cd /vision/build
rm -rf *

# Enable CUDA support
cmake -DCMAKE_PREFIX_PATH=/libtorch -DWITH_CUDA=on ..
make -j8
make install
```

# Build boomer_localization (Inside Docker)
```bash
cd /your-boomer-tool-path
mkdir build
cd build
cmake ..
make -j8
```

# Pretrained Model and Test Data
Download from:

Model: https://drive.google.com/drive/folders/1zWE2_VW2Dwd2bMmDApECs3VcoYkRMuAX?usp=sharing
Test data: https://drive.google.com/drive/folders/1aAUneEyRSL1V0ZQkosvgni-erFZubJ9-?usp=drive_link

Place:

- .pt model files → into the models/ folder

- Test data → into the data/ folder

# Run Matching Example 
```bash
cd build
../scripts/test_command.sh
```
This command will:

- Match two point clouds

- Stitch them together

- Save the stitched result

The stitched point cloud will contain the two inputs in different colors.

# TEASER++
Repository: https://github.com/MIT-SPARK/TEASER-plusplus.git
Follow the official installation instructions.

# FAQ:

## Torchvision CUDA Error in Docker

If you see:
```
terminate called after throwing an instance of 'std::runtime_error'
...
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

This usually means torchvision was not rebuilt with CUDA support.
Rebuild it:
```bash
cd /vision/build
rm -rf *

cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DWITH_CUDA=on ..
make -j8
make install
```

## CUDA17 Dialect Error
```
Target "xxx" requires the language dialect "CUDA17"
```
Cause:

CMake version too old

Solution:

Use CMake >= 3.18

Reference:
https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_STANDARD.html

## OpenCV / PCL Linking Error
Example:
```
undefined reference to `cv::imread(std::string const&, int)'
```

Possible causes:

    - ABI mismatch (cxx11 vs pre-cxx11)

    - Incorrect OpenCV/PCL build configuration

Reference:
https://medium.com/mlearning-ai/pytorch-c-4-using-torchvision-models-f12bd25f4744

Avoid building OpenCV/PCL with:
```
-DGLIBCXX_USE_CXX11_ABI=0
```
unless absolutely necessary.

## Conflicting CUDA Installations
```
Found two conflicting CUDA installs:
V11.8.89 in '/usr/local/cuda-11.8/include'
V11.8.89 in '/usr/local/cuda/include'
```
Check:
```bash
echo $CUDA_HOME
echo $CUDA_PATH
```
Ensure they match your compiler:
```
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)`
```
If necessary:
```
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=/usr/local/cuda-11.8
```