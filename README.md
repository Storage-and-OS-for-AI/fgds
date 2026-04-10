## Abstract
fgds is a fork of [phoenix](https://github.com/nicexlab/phoenix), a refactored I/O stack for GPU Direct Storage without phony buffers. fgds extends phoenix with the following key enhancements:

* **Enhanced Availability**: Fixed kernel module loading failures caused by BAR address conflicts and memory manager bugs
* **Improved Compatibility**: Added support for both x86 and ARM environments, adapted to kernel 6.6, and enabled multi-GPU scenarios with specified GPU lists
* **Ecosystem Integrations**:
  * Added support for kvcache offloading through vllm lmcache backend, significantly boosting vllm token throughput compared to GDS
  * Integrated fastsafetensors for accelerated model loading, substantially reducing Qwen-32B model loading latency on H20 GPU compared to GDS
  * Implemented the PyTorch API of fgds and optimizing the performance of checkpoint save and load 

The phoenix has been accepted for [SC'25](https://doi.org/10.1145/3712285.3759862). 

This documentation explains how to build, configure and use this I/O stack.

## Directory structure
```
fgds
├── benchmarks   # artifact evaluation files
├── example      # example for how to use fgds
├── libfgds      # user library for fgds
├── module       # kernel module for fgds
├── scripts      # test scripts
└── python       # python api for fgds
```

## How to Build

### Soft Environment
* OS: Ubuntu 22.04.2 LTS/KylinOS Server V11
* kernel: Linux 6.1.0-rc8/6.6.0
* NVIDIA driver: 550.54/570.124.06
* OFED driver: 24.10
* CUDA: 12.4/12.8
* CMAKE: 3.18+
* GCC: 12.0+

### Hardware Environment
* PCIe P2P DMA supported
* GPU device and NVME SSD should be installed in the same PCIe root complex and not through expansion cards
* Above 4G Decoding needs to be ENABLED in the BIOS
* Disable ACS
* IOMMU disabled
  * The system's IOMMU should be disabled for ease of debugging
    In Intel Systems, this requires disabling Vt-d in the BIOS
    In AMD Systems, this requires disabling IOMMU in the BIOS
    In ARM Systems, this requires disabling SMMU in the BIOS
  * The iommu support in Linux must be disabled too.

### 1. NVIDIA GDS

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo bash cuda_12.4.0_550.54.14_linux.run
# select nvidia-fs option and choose open driver
```

### 2. MLNX_OFED Driver
```shell
sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support --dkms --skip-unsupported-devices-check
sudo update-initramfs -u -k `uname -r`
sudo reboot
```
### 3. NVMe-of/NFS
#### 3.1 NVMe-of
```shell
cd scripts
sudo bash nvme_of.sh <target|initiator> <setup|cleanup>
```
#### 3.2 NFS
```shell
cd scripts
sudo bash nfs.sh <server|client>
```
### 4. Fgds and Benchmarks
```shell
mkdir -p build
cd build && cmake ../
make -j 
```
Note: this will compile all the benchmarks including the kernel module
## How to Use
### 0. Edit the config.json
First, in the config.json file located in the project's root directory, specify which GPUs to use. If `use_all_gpus` is set to `true`, it indicates that all GPUs on the machine will be used. If `use_all_gpus` is set to `false`, it indicates that only the GPUs specified in the `gpuids` array will be used.

The priority of `use_all_gpus` is higher than that of `gpuids`. That is, as long as `use_all_gpus` is set to `true`, all GPUs will be used regardless of the content of `gpuids`.
### 1. Install Kernel Module

```shell
sh scripts/load_fgds.sh 
```
This script will utilize the GPU specified in config.json
If you directly run sudo insmod build/module/fgdsfs.ko, it will default to using only GPU0.

Note: must run `nvidia-smi` to `modprobe` nvidia driver before install fgds kernel module.
### 2. Example for Using libfgds
We have provided a simple example to illustrate how to program using libfgds
see [example/example.cc](./example/example.cc)

### 3. Fgds and python api
see [python/README.md](./python/README.md)

### 4. Evaluation Procedure
We provide some scripts to execute the evaluation procedure.

Note: make sure to update the paths in the scripts.
#### 4.1 Faster Reproduction. 
We have integrated all experiment scripts and provided a Python script. 
Users can run the corresponding experiment by specifying the artifact parameter. This script will also print all the corresponding execution commands.
Before running this Python script, users need to set the variables `file_path`, `nvmeof_file_path`, and `model_dir` to the paths specific to users’ own environment. 
All results will be stored in the `fgds/sc25/results` directory.

```shell
cd fgds/sc25
# `all` will run table3 and fig 3 ~ 12
sudo python run_all_benchmarks.py --artifact all
```
In addition, we also provide individual scripts for each experiment as follows:
#### 4.2 Breakdown
```shell
cd scripts && sudo bash breakdown.sh
```
#### 4.3 I/O Performance
```shell
cd scripts
# see micro.py for detail
sudo python micro.py <0|1> <0|1|2> 0
```
#### 4.4 End-to-End Performance
```shell
cd build/
sudo bin/end-to-end <file_path> <io_size> <mode>
```
#### 4.5 KVCache Loading
```shell
cd scripts
sudo bash kvcache.sh
```
#### 4.6 Model Loading
```shell
cd scripts
sudo python load_safetensors.py
```