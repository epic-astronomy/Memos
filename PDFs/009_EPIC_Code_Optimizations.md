# EPIC MEMO #009

<p  style="text-align:center; font-size: 2em">ARIZONA STATE UNIVERSITY<br>TEMPE, ARIZONA 85287</p>
<p  style="text-align:center; font-size: 1.5em">EPIC Code Optimizations-Part I</p>
<p  style="text-align:center; font-size: 1.5em">Karthik<br>Feb 4
, 2023</p>

## Introduction

EPIC sky-imager's first iteration is a python-based program powered by the Bifrost framework. Each payload passes through a series blocks in a pipeline that transform the data using CPU-based and/or GPU-based operations. These blocks communicate using ring-buffers, which implement a locking mechanism for appropriate storage and retrieval by these blocks. 

The GPU part of the pipeline mainly focuses on applying the EPIC algorithm to gulps (or a series of packet sequences) of FEngine data to produce images. The relevant operations are carried out in the `MOFFCorrelatorOp` block. Optionally, GPUs can also be used in averaging the channels before imaging them in the `DecimationOp` block.

 The CPU part of the pipeline performs the following tasks using independent blocks or operators:
- `DecimationOp`: Decimate the channels if need be (or average them on a GPU).
- `SaveImageOp`: Post-process the image for further storage.


The main intent of this work is to be able to produce sky-images covering the full bandwidth and all the polarizations. Below I will describe the profiling results and metrics of the current GPU blocks, and I will discuss the new optimizations that allow us to obtain full bandwidth and full polarization images. 

## GPU Profiling
Optimizing GPU kernels require a proper identification of the associated bottlenecks through profiling. Below I provide a brief introduction to the concepts used in GPU programming, which aid in interpreting the bottlenecks, followed by GPU profiling results of the EPIC code.

### CUDA Primer
EPIC instances rely on NVIDIA GPUs (A4500 at the time of writing this memo) for parallel processing. Programs use the C/C++ interface provided by the Compute Unified Device Architecture (CUDA) programming model to parallelize scalar programs. Its compiler provides all the necessary abstractions to leverage massive parallelism built into the CUDA programming model. CUDA provides a thread and memory hierarchy to simplify GPU programming. Its thread hierarchy can be described as follows:
* __Blocks__: A collection of threads with 3D-indexing support. Each thread executes the same kernel and each block can have up to 1024 active threads at a time.
* __Grid__: A collection of blocks with 3D-indexing support (see Figure 1). Each grid can be launched with up to 2^31-1 blocks, however, the maximum number of active blocks in the grid depends on multiple variable, for instance, the type of GPU, available memory, among others.
* __Warps__: A collection of 32 threads with consecutive indices bundled together and executed on a single CUDA core. The warp dimension is independent of the block and grid sizes. The cores are grouped together to form a streaming multiprocessor (SM).
* __Cooperative Groups__: In addition to the above-mentioned collectives, CUDA also offers a collective for selective grouping of threads known as cooperative groups. Here a group can be a collection of threads within a block or a collection of blocks within a grid (requires devices with compute capability of 9.0, or greater). For instance, to perform convolutional antenna gridding, threads within a block can be grouped into "tiles" and each tile can compute all the grid elements of a single antenna in parallel (see section 4.3).

<p align="center">
  <img src="https://user-images.githubusercontent.com/4162508/214206097-b8887ecf-6c04-4b69-b0e1-cb821ff84cd7.png" />

  __Figure 1__: Grids, Blocks and Multiprocessors. A block is a collection of threads and a grid is a collection of blocks with 3D indices. Threads with consecutive indices are bundled together into warps of 32 threads and are excecuted on individual CUDA cores. NVIDIA GPUs consist of SMs each with several CUDA cores.
</p>
CUDA programming model also provides a memory hierarchy accessible to the threads (see Figure 2). Each thread has private local memory (registers) and each thread block has a shared memory accessible by all its threads, which can be used as a scratchpad  or L1 memory. Blocks can also access each other's shared memory in devices with compute capability 9.0. All threads can access the global memory through the L2 cache. The shared memory size is extremely limited and 

<p align="center">
  <img src="https://user-images.githubusercontent.com/4162508/214466324-8788cb76-033e-4c98-b803-ff146d053d1a.png" />

  __Figure 2__: CUDA memory hierarchy with example memory sizes for NVIDIA A100.
</p>


In addition, there exists read-only memory accessible to all the threads, namely, constant and texture memory. These memory spaces are optimzied for different memory access types. For example, the texture memory space is optimized for spatially correlated access patterns that can be useful in gridding convolution.

Bottlenecks exist in GPU codes either due to inefficient memory accesses (bandwidth-bound) or due to GPU's inadequate computation power (compute-bound). Below I describe profiling results of the EPIC's GPU code that can be interpreted using the above described characteristics, and I will detail the optimizations are made to make the kernels compute-bound.

### Profiling
I used [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) to measure the overall performance (or system-level profiling), and [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) to measure the kernel metrics. See [Pearson (2020)](https://www.carlpearson.net/pdf/20200416_nsight.pdf) for an overview on these tools. For both tools, I ran EPIC on the main EPIC server using live data from the LWA-SV telescope. The `MoFFCorrelatorOp` block was profiled for 50 s with the following configuration:
- Gulp size 40 ms (or `--nts 1000`)
- Single polarization
- Image size 64x64
- Image resolution 1.8 deg
- Accumulation time (80 ms)
- Channels 90

Run command:
```bash
LD_PRELOAD="libvma.so" VMA_RX_POLL=1000 VMA_INTERNAL_THREAD_AFFINITY=0 VMA_RX_PREFETCH_BYTES=128\
 VMA_THREAD_MODE=0 VMA_MTU=9000 VMA_TRACELEVEL=0 numactl --cpubind=1 --membind=1\
 ./LWA_bifrost.py  --addr 239.168.40.16 --port 4015 --channels 90\
 --imagesize 64 --imageres 1.8  --accumulate 80 --nts 1000 --singlepol --duration 50
```
The EPIC server is powered by an Intel® Xeon® Silver 4210 CPU with a CPU clock speed of 2.20GHz and the profiling was carried out on an NVIDIA RTX 2080 Ti GPU.
### Results
Figure 3 shows a screenshot from the Nsight Systems UI depicting the sequence of operations that are carried out in the correlator block. The typical processing times for a 40 ms gulp are as follows. 
```
- Memset the cross-multiplied data to 0 (once per image)   [5.0  ms]
Loop until accumulation>=80 ms
 - Load antenna (or un-gridded) data                       [1.9  ms]
 - Memset the gridded data to 0                            [5.0  ms]
 - Grid the data (VGrid kernel)                            [2.7  ms]
 - Inverse FFT the gridded data (vec_2d_fft kernel)        [11.0 ms]
 - Cross multiply (XGrid kernel)                           [16.6 ms]
- Combine all images in the gulp (reduce kernel)           [5.0  ms]
-                                              Total time: [42.2 ms]
```
The total time to image two 40 ms gulps is therefore `42.2 * 2 + 5 = 89.4 ms`. The final image has dimensions of [2 (real, complex), npol, nchan, grid_size, grid_size]. 

The profiling data is available in the `profiling/EPICv0.1` folder of the [epic-stream-processor](https://github.com/epic-astronomy/epic-stream-processor) repo. See the [user guide](https://docs.nvidia.com/nsight-systems/pdf/UserGuide.pdf) for additional information on Nsight Systems UI. 
> Notice the ~4 ms gap between reduce kernel and `D2H` copy of the reduced image. I'm unsure what is causing this, but I presume it is the time required to acquire a lock on the output ring buffer, which is an input to the `SaveOp` block. Although it could be a gap introduced by the profiler, it is unlikely for the gap to consistently exist only between the _reduce_ and the _copy_ steps.

![Screenshot from 2022-11-23 13-28-46](https://user-images.githubusercontent.com/4162508/203640587-3e21ed17-5943-466a-a822-a0ad8e05fb1f.png)
**Figure 3**: Screen shot of Nsight Systems UI. The timeline is zoomed-in to show a single sequence of imaging operations.

Although the FFT kernel runs in a different stream than VGrid and XGrid, they do not overlap. Furthermore, the FFT and the XGrid kernels appear to be the hotspots in imaging. To see if there is room for any improvement, let us look at the profiling results from Nsight Compute. I used the same options provided above to run the EPIC imager. See [Subrahmaniam (2021)](https://ericdarve.github.io/cme213-spring-2021/Lecture%20Slides/CME213_2021_CUDA_Profiling.pdf) for a review on using Nsight Systems and Compute to optimize GPU kernels. Below I will describe the main metrics for each kernel.

### VGrid
This kernel grids the antenna data. The default settings provide a support size of a single cell. Put another way, each antenna only contributes to the cell it falls in. Figure 4a shows that the kernel is bandwidth-bound and that memory is more heavily utilized than compute SMs. That means, for a given unit of memory bandwidth utilization, non-optimal work is being done.

![image](https://user-images.githubusercontent.com/4162508/203662798-ee8b27be-e1ab-4586-bd5b-39525d4d9330.png)
**Figure 4a:** Screen shot of GPU throughput for VGrid from the Nsight Compute UI. This kernel achieves ~12% of the max compute throughout while ~72% memory throughput, making it memory bound.
![image](https://user-images.githubusercontent.com/4162508/203663075-b25430e3-e505-438c-8099-3aac853e4dec.png)
**Figure 4b:** Screen shot of Performance vs. Arithmetic Intensity for VGrid from the Nsight Compute UI. The first and second square markers from bottom indicate the peak double and single precision performances (or _rooflines_), respectively. This kernel achieves ~12% of the max compute throughout while ~72% memory throughput, making it bandwidth bound.

The first roofline in Figure 4b indicates the peak double-precision performance, while the second one indicates the peak single-precision performance. The achieved performance is shown as a filled-green circle that is about half of the peak double-precision performance. Note that the kernel achieves nearly 100% warp occupancy. That means the kernel keeps the GPU fully occupied during the computation, however, does not achieve a high FLOPS/byte. It is because the kernel spends most of the time accessing data from the global memory.
<!--
![image](https://user-images.githubusercontent.com/4162508/203663666-c18a9fcf-7a46-42fd-a146-a3f8ce247c2e.png)
>TODO: Improve global memory accesses in VGrid. However, note that this step only takes about 7% of the total process time, and hence should be prioritized below the rest.
-->

### vector_2d_fft
This kernel preforms inverse FFT on the gridded antenna data to produce sky images. The FFT kernel is also bandwidth bound similar to the VGrid kernel. The FLOPS/bytes (3.06) is higher than the expected double-precision roofline (0.57) suggesting the amount of work done per byte is near the optimal value.

![image](https://user-images.githubusercontent.com/4162508/203670528-5e20494f-bebd-4914-98f9-cab06de118e8.png)
**Figure 5a:** Same as Fig. 2a but for `vector_2d_fft`

![image](https://user-images.githubusercontent.com/4162508/203671353-75de93f1-c126-4dc8-b050-6e064653fb82.png)
**Figure 5b:** Same as Fig. 2b but for `vector_2d_fft`

The theoretical warp occupancy (fraction of the total available active warps occupied) is, however, only 50%. It is because the shared memory required by all the active warps is greater than the available memory that, in turn, restricts the warp occupancy to 50%.

<!--
![image](https://user-images.githubusercontent.com/4162508/203671257-a3942405-d9c0-405a-9711-c64e45b5d043.png)
![image](https://user-images.githubusercontent.com/4162508/203671871-60470b60-f028-473d-b5fd-700eb7328d60.png)

Furthermore, the above figure shows that the warp occupancy can be improved by increasing block size. 
-->

### XGrid
This kernel is responsible for producing the cross-multiplied products (XX, YY, XY*, YX*). In this case, only the XX product is generated. Although XGrid is bandwidth bound similar to VGrid and FFT, it has as >80% bandwidth usage as shown in Figure 6a, indicating efficient memory access patterns. However, its performance is still below the double precision roofline (see Fig. 6b).

![image](https://user-images.githubusercontent.com/4162508/203678892-2584babd-b108-4288-ab92-55ccd2460782.png)
**Figure 6a:** Same as Fig 2a but for `XGrid` 
![image](https://user-images.githubusercontent.com/4162508/203678979-6238ebe0-8601-4a4b-b59d-e3378405a12e.png)
**Figure 6b:** Same as Fig 2b but for `XGrid` 

## Discussion
The profiling results clearly indicate that all the kernels used in the correlator block are bandwidth bound. It is primarily due to each kernel using the global memory as their workspace: 
1. VGrid requires large writes to the global memory. For instance, the write-size for a 64x64 image with a gulp size of 1000 sequences each with 90 channels and a single polarization is ~2 GiB. 
2. The FFT kernel although performs above the double precision roofline, it must fetch the gridded data and write iFFTed data back to the global memory.  
3. The XGrid kernel must again fetch the iFFTed data from the global memory to derive cross multiplication products and write them back to the global memory.

Furthermore, for each accumulation, initializing the gridded and cross multiplied data blocks consumes ~19% of the total accumulation time. These bottlenecks can be eliminated by coalescing these kernels and performing computations using on-chip memory (registers and shared memory), which is about ten times faster than global memory.

A new package called [cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/index.html), which is distributed with MathDx, allows us to perform FFT operations completely using registers and shared memory. Although using on-chip memory eliminates the need to perform computations using global memory, on-chip memory is highly limited (e.g., ~100 KiB per SM in A4500) thereby limiting the size of ouptut images. For instance, a 64x64 image with complex double precision occupies 64 KiB. That means only a single block can be run per SM that leads to <50% warp occupancy thereby increasing the overall runtime. However, with half (16 bit) precision, we can fit two 64x64 images in a 32 KiB shared memory block. This permits us to generate a single image with both polarizations within the same block. On RTX 2080 Ti, it takes about ~13 ms to perform iFFT on a 1000-sequence glup with 132 (3.3 MHz) channels for both the polarizations, which is similar to the time taken to iFFT an identical gulp but with only 90 channels and a single polarization. Below I describe how the VGrid and XGrid kernels can be combined with cuFFTDx to perform gridding and iFFT followed by cross multiplication within a unified kernel completely using on-chip memory.


### FFT
Although gridding is performed before FFT, using cuFFTDx constrains the performance of gridding and cross multtiplication. Hence, I will describe the FFT part of the unified-kernel before discussing the former two.

The current version (v1.1.0) of cuFFTDx only supports 1D fourier transforms. Hence, 2D iFFT is performed by computing row-wise iFFT followed by column-wise iFFT. cuFFTDx reads input data from thread registers and uses shared memory as a workspace to compute FFT. The output is stored back again to the registers. It is mainly configured using four parameters:
* FFT size (Size of FFT along one dimension)
* Precision
* Registers per thread (Number of FFT elements that will be computed in each thread)
* FFTs per block (Number of `FFT size` FFTs per block)

To perform iFFT on a single square image, we must set `FFT size` and `FFTs per block` each to the side of the image. The number of registers per thread, which must be a power of 2, and precision can be varied to find the optimal set of parameters. These four parameters determine the block size and the shared memory required for the FFT operation. For example, to iFFT a 64x64 image with 32-bit precision, we must set `FFT size` and `FFTs per block` to 64. If each thread computes 8 FFT elements, the kernel block must be lauched with dimensions of `(8, 64, 1)` and a shared memory of 64 KiB, as shown schematically in Figure 7.

![fft_block_scheme](https://user-images.githubusercontent.com/4162508/216741704-3ad440ee-0b86-4ef8-9d6e-0af50d081dfa.jpg)
**Figure 7:** Distribution of FFT elements in a 2D thread block for a 64x64 image with 8 elements per thread. The numbers inside a thread indicates the pixel indices for that row. Each element is a complex number for single and double precisions while it is a set of two complex numbers for half-precision arranged in RRII layout. Each row computes all the FFT elements for the corresponding row in the output image. In case of half-precision, the FFT elements of both the images are computed simultaneously.

With a 64 KiB shared memory requirement, only one block can be run per SM (on devices with compute capability<9.0), which reduces the GPU throughput. Although the memory size for FFT can be reduced by increasing the number of thread registers, we will still require 64 KiB to transpose the image for column-wise FFT. Hence, the unified kernel must be able to fit an entire image within the shared memory. Furthermore, because each block can only image one polarization, global memory is required to store and fetch images to generate cross polarization products, which introduces additional latency. These two problems can be addressed by using half-precision (16-bit float) where each block can accomodate two images in 32 KiB shared memory and all polarization products can be generated using on-chip memory. 

The memory layout for half precision differs from 32-bit and 64-bit precisions. While each register holds one complex number in the latter two, two complex numbers are batched together into a single register with the former in an RRII layout. That means we can use the same set of registers to hold two images at a time. This allows us to compute iFFT and generate cross polarization products in a single operation.

Using half-precision alleviates the shared-memory requirements, however, it introduces problems of floating point over or underflows. The voltage data although arrives as unsigned 4-bit complex numbers, gridding can produce values upto a few tens and inverse FFT can result in pixel values overflowing 65504, which is the largest representable magnitude in half-precision. Hence, the values are normalized by the image size (N) after the row-wise iFFT to prevent this overflow. The normalization factor is only $\frac{1}{N}$ as opposed to the conventionally used factor of $\frac{1}{N^2}$ to prevent underflow. For example, $\frac{1}{N^2}\approx 2.4 * 10^{-4}$ for a 64x64 image, which is close $\approx 6.4 * 10^{-5}$, the smallest magnitude representable in half-precision. That means any normalization with $\frac{1}{N^2}$ on values less than $\approx$ 0.24 would reult in an underflow.  

### Gridding  
Previous [optimization](https://github.com/epic-astronomy/Memos/blob/master/PDFs/003_Romein_Optimization.pdf) to the gridding involved processing each antenna on a separate thread (block size=256) and atomically adding the gridded data back to the global memory with the support size of a single grid cell (pill box gridder). However, the block dimenions, which are determined by cuFFTDx, may not always equal the number of antennas. Hence, in the unified kernel, the threads are divided into _tiles_ (or groups) of size `support x support` using the `cooperative_groups` (CG) functionality provided by CUDA. With CG, the tile size is restricted to powers of 2. Each tile computes all the grid values of a single antenna and atomically adds to the grid that is initially stored in the shared memory and then tranfered to the thread registers. CG provides functionality to determine the 1D index (or rank) of each thread within the tile group. We can convert this index into a 2D position on a local grid relative to the antenna that can then be used to determine its nearest grid point on the UV plane (see Figure 8). The following pseudo-kernel demonstrates this algorithm.
```c++
for(ant=tile_rank,ant<N,ant+=ntiles){
  antx = antenna_pos[ant].x
  anty = antenna_pos[ant].y
  //Determine antenna's local grid point 
  //In each tile, thread_rank goes from 0...T-1 where T=support*support
  v = int(thread_rank/support) - support * 0.5 + 0.5
  u = thread_rank  - int(thread_rank/support)*support - support * 0.5 + 0.5

  //Check if the grid point falls inside the UV grid after translating it to the antenna's position

  //If yes
  //calculate antenna's offset from the UV grid point
  dx = abs(int(antx + u) + 0.5 - antx)
  dy = abs(int(anty + v) + 0.5 - anty)
  
  //fetch the scaling factor using normalized coordinates
  scale = texture_2D(gcf_kernel, dx/(0.5 * support), dy/(0.5 * support))

  //use the scaling factor and atomically add the appropriate voltage 
  //to the UV grid
  //The corresponding grid coordinate's data index is given by
  U=int(antx + u)
  V=int(anty + V)
}
```
![gridding](https://user-images.githubusercontent.com/4162508/216746331-03467423-d2ba-420b-bae1-5e2a4c8ab0ef.jpg)
**Figure 8:** Gridding examples with a support size of 4 cells. In the new gridding approach, each cell in the antenna's kernel is assigned a thread from the tile group. These threads make atomic updates to the grid in the shared memory. The left panel shows the gridding work distribution when the antenna location falls in between the UV grid points (blue-dots). In this case, each thread computes the grid value closest to its cell's center. For example, thread 0 computes the grid value for (1,0). The right panel shows the gridding work distribution when the antenna location falls on the UV grid points. In this case, because multiple grid points fall on the boundary of each cell, only the grid point closest to the bottom-left corner of each cell is considered. No update is made for grid points falling on the boundary of the kernels (grey dots).

 The Gridding Convolution Function (GCF) is identical for all the antennas and is a zero-order prolate spheroid, stored as a 2D texture. Because the kernel is symmetric about the origin, the texture only contains the first quadrant of the kernel. In the unifed kernel, the gridding kernel resolution is set to 32 elements or the GCF is pre-computed on a 32x32 grid. NVIDIA GPUs have specialied hardware to fetch textures at non-integral indices. This allows us to determine the scaling factor for each grid point located at arbitrary distances from the antenna. Finally, it is possible to use different GCFs for each antenna and polarization by extending the 2D texture along the third dimension without introducing any additonal latency.


### Cross Multiplications
Due to the RRII layout provided by half-precision in each register, the cross multiplication products (XX, YY, XY*, YX*) for each pixel can be derived within the thread without requiring any global memory reads. To avoid atomic writes to the global memory, each channel is imaged in its own dedicated block. This will also eliminate the need to initialize the output memory block to 0 because the first write to the memory will be assignment and the rest will be in-place additions. Furthermore, because each accumulation can be an average of hundreds to thousands of individual image sequences, the output image block is assigned floating (32-bit) precision to prevent over and underflows.

## Current State and Way Forward
With the unified-kernel, on `NVIDIA A4500`, which has 56 SMs, it takes ~30 ms to generate a 64x64 sky-image with 112 (2.8 MHz) channels and both polarizations although with a support size of only 2 cells. Moreover, the total memory requirement including constant and texture memory is less than 200 MiB. The bandwidth is limited because of the shared memory constraint of 32 KiB per block that allows only 112 simultaneous active blocks (2 blocks per SM) on the GPU. Any GPU with an SM count>66 (e.g., RTX 2080 Ti; see also table 1) would allow us to generate images with the full bandwidth of 3.3 MHz.

The image size is also limited to 64x64 due to shared memory constraints. Although a single 128x128 image can fit in the shared memory, it will restrict the maximum active blocks per SM to one. Because the FFT is performed row-wise followed by column-wise, the FFT can be split to multiple blocks where each block performs row-wise FFT, transposes the image and exchanges column-wise data from adjacent blocks.  This inter-block communication can be achieved using clusters of blocks with distributed shared memory. Although this method removes global memory accesss, it requires GPUs with compute capability of at least 9.0. At the time of writing this memo, `H100` is the only GPU with this capability. Alternatively, two GPUs can be run in parallel, one for each polarization to produce 128x128 images at near full bandwidth. However, it will introduce a global memory dependency to generate XY* and YX* images leading to larger run times. Profiling must be carried out to determine the impact of global memory writes in the 128x128 imaging mode. Below I provide a list of potential GPUs capable of powering EPIC imagers at LWA, Sevilleta.


**Table 1:** List of potential GPUs for EPIC imagers
|GPU|SM count | Shared Memory per SM (KiB)|Image size|Achievable bandwidth<sup>a</sup> (Channels/MHz)|Polarization|Base [Boost] Clock (MHz)|Compute capability|Bus interface|Estimated price
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
RTX 2080 Ti| 68 | 64 | 64x64<br>128x128 | 132/3.3 MHz<br>68/1.65 MHz| Dual<br>Single|1350 [1545]| 7.5|PCIe 3.0 x16 |$1k
RTX A4500| 56 | 128 | 64x64<br>128x128 | 112/2.8 MHz<br>56/1.4 MHz|Dual<br>Single |1050 [1650]|8.6|PCIe 4.0 x16| $1.5k
RTX 3090 Ti | 84 | 128 | 64x64<br>128x128 | 132/3.3 MHz<br>84/2.1 MHz|Dual<br>Single | 1560 [1860]|8.6|PCIe 4.0 x16|$1.6k
RTX 4090 | 128 | 128 | 64x64<br>128x128 | 132/3.3 MHz<br>128/3.2 MHz| Dual<br>Single | 2235 [2520]|8.9|PCIe 4.0 x16|$1.8k
RTX 4090 Ti | 142 | 128 | 64x64<br>128x128 | 132/3.3 MHz<br>132/3.3 MHz| Dual<br>Single | 2235 [2520]|8.9|PCIe 4.0 x16|$2k<sup>#</sup>
A100 PCIe 40 GB| 108 | 192| 64x64<br>128x128 | 132/3.3 MHz<br>108/2.7 MHz| Dual<br>Dual |765 [1410] |8.0|PCIe 4.0 x16|$11k
H100 PCIe | 114 | 256 | 64x64<br>128x128 | 132/3.3 MHz<br>114*/2.85 MHz| Dual<br>Dual | 1095 [1795]|9.0|PCIe 5.0 x16|$35k

<sup>a</sup>Estimated by assuming each channel will be imaged on a dedicated block.

<sup>#</sup> Estimated price only. Set to be released in late 2023.

\* Due to support for distributed shared memoy, it may be possible to acheive full bandwidth by splitting the FFTs among multiple blocks.








# References
- [Carl Pearson 2020, Using Nsight Compute and Nsight Systems](https://www.carlpearson.net/pdf/20200416_nsight.pdf)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/pdf/UserGuide.pdf)
- [Akshay Subrahmaniam 2021, WHAT THE PROFILER IS TELLING YOU: OPTIMIZING GPU KERNELS](https://ericdarve.github.io/cme213-spring-2021/Lecture%20Slides/CME213_2021_CUDA_Profiling.pdf)
