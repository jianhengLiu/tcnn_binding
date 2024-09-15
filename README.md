# tcnn_binding

```bash
# LibTroch: https://pytorch.org/get-started/locally/
  # Tested on:
  # - CUDA/Torch: 11.8/2.0.0
  wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
  # for other cuda version use the following command to download the corresponding version of libtorch(cuda11.3 for example):
  # wget https://download.pytorch.org/libtorch/nightly/cu113/libtorch-cxx11-abi-shared-with-deps-latest.zip
  unzip libtorch-cxx11-abi-shared-with-deps-*.zip
  echo "export Torch_DIR=$PWD/libtorch/share/cmake/Torch" >> ~/.zshrc
```

CMakelist.txt
```
# tcnn_binding
add_subdirectory(submodules/tcnn_binding)
include_directories(
  submodules/tcnn_binding
  submodules/tcnn_binding/submodules/tiny-cuda-nn/include
  submodules/tcnn_binding/submodules/tiny-cuda-nn/dependencies)
```
