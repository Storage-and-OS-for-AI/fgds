# FGDS 是基于开源项目[phoenix](https://github.com/nicexlab/phoenix)研发的, 是对NVIDIA GDS（GPU Direct Storage）技术的优化 
Phoenix 是[SC'25](https://doi.org/10.1145/3712285.3759862)上的一篇顶会论文. 

Phoenix是在GDS基础上做的性能优化，GDS在主机内存中需要使用phony buffer与
Linux内核交互，占用了内存空间，并且申请和释放phony buffer都需要时间，消耗了IO时延，此外，使用GDS也需要安装MLNX_OFED内核驱动套件来增强NVME内核模块，增加了部署复杂度。

Phoenix利用Linux ZONE_DEVICE内存映射服务，在IO路径上完全消除了phony bufer，解决了下列核心问题：

1）消除了GDS申请和释放phony buffer的耗时，降低了IO传输时延；

2）避免了phony buffer的内存占用，提升了IO带宽；

3）不再像GDS那样需要安装MLNX_OFED内核驱动套件，降低了部署复杂度。

这篇文档是关于怎样使用fgds的。 
 
 
## 使用步骤
### 0. 安装NVIDIA内核模块和CUDA依赖
fgds的运行依赖NVIDIA内核模块，示例代码会调用到CUDA API，因此需要先安装这些。

可通过 https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux 进行安装。

### 1. 编辑config.json
首先，编辑项目根目录下的config.json，来指定要绑定哪些GPU。该配置文件中包含use_all_gpus和gpuids两个字段，如果use_all_gpus为true，则表示绑定所有GPU，如果use_all_gpus为false，则表示绑定gpuids数组中指定的GPU。gpuids是整形数组，里面可以指定要绑定的gpu的id，如0,1等。
use_all_gpus字段的优先级高于gpuids，即只要use_all_gpus为true，则就是会绑定所有GPU，而无论gpuids的值为什么。

### 2. 运行脚本
```shell
sh scripts/load_fgds.sh 
```
这个脚本会绑定config.json中指定的GPU。如果不通过该脚本，直接通过insmod方式安装fgdsfs.ko, 则默认是只绑定GPU0。

如果fgds内核模块不存在，该脚本会在/opt/fgds/module目录先编译内核模块，然后再加载。


注意: 

1）加载fgds内核模块依赖NVIDIA内核模块先加载。所以先运行 `nvidia-smi`来确保NVIDIA相关内核模块已经加载。 

2）在加载fgds内核模块时，会把GPU的PCIE Bar物理地址映射进内核的ZONE_DEVICE类型虚拟内存。如果一块GPU的PCIE Bar物理地址中，有部分物理页的PAT(内存缓存策略)存在冲突且无法被跳过冲突地址，则该GPU无法被fgds内核模块绑定，可以通过dmesg或/var/log/messages查看具体报错信息。

这种情况一般是GPU的PCIE Bar地址已经被某些进程使用了，存在冲突，需要先释放这些被冲突的地址，才能用fgds内核模块绑定该GPU。具体方式为运行sudo fuser -v /dev/nvidia*来查看是哪些进程使用了该GPU，例如对于GPU0，运行sudo fuser -v /dev/nvidia0 来查看。

3）当在config.json中指定了绑定多块GPU时，如果部分GPU由于上文所述的PCIE Bar地址存在PAT冲突而无法被fgds绑定，则fgds会跳过这些GPU，只绑定能成功绑定的GPU，此时fgds内核模块仍然会加载成功，用户可以使用这些已绑定的GPU。在卸载fgds内核模块时，fgds也只会释放这些已绑定的GPU的资源。

只有在config.json中指定的所有GPU都存在PAT冲突而都无法被fgds绑定，fgds内核模块才会加载失败。

### 3. 使用libfgds的示例
我们提供了一个简单的示例程序来展示如何使用libfgds api，请参考 [example/example.cc](./example/example.cc)

### 4. fgds的python api
请见 [python/README.md](./python/README.md)

### 5. 性能测试程序
我们提供了性能测试程序，来测试fgds、gds、和传统方式（GPU读写NVME盘时，由CPU参与进行内存拷贝，例如GPU读NVME盘，先由CPU读NVME盘到内存，再拷贝到显存）的性能。

使用方法如下：

编辑scripts/micro.py，在io_sizes数组中指定要测试的blocksize，单位为KB。
运行脚本，在命令行入参中指定要运行的GPU卡的id，要测试的方法(fgds/posix/gds)

例如

```
#### I/O Performance
```shell
cd scripts
python /root/pkl/fgds/scripts/micro.py  0  fgds sync nvme  /data/10GB_ddrand  
表示在gpu0号卡上测试fgds，使用NVME盘/data/10GB_ddrand作为测试文件（用fgds读写该文件）

python /root/pkl/fgds/scripts/micro.py  0  posix sync nvme  /data/10GB_ddrand 
表示在gpu0号卡上测试传统方式（经过CPU内存拷贝的方式）的性能，使用NVME盘/data/10GB_ddrand作为测试文件
```