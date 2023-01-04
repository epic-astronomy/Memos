# EPIC MEMO #009

<p  style="text-align:center; font-size: 2em">ARIZONA STATE UNIVERSITY<br>TEMPE, ARIZONA 85287</p>
<p  style="text-align:center; font-size: 1.5em">EPIC Code Optimizations-Part I</p>
<p  style="text-align:center; font-size: 1.5em">Karthik<br>Jan 2, 2023</p>

## Introduction

EPIC sky-imager's first iteration is a python-based program powered by the Bifrost framework. Each payload passes through a series blocks in a pipeline that transform the data using CPU-based and/or GPU-based operations. These blocks communicate using ring-buffers, which implement a locking mechanism for appropriate storage and retrieval by multiple producers and receivers. 

The GPU part of the pipeline mainly focuses on applying the EPIC algorithm to gulps (or a series of packet sequences) of FEngine data to produce images. The relevant operations are carried out in the `MOFFCorrelatorOp` block. Optionally, GPUs can also be used in averaging the channels before imaging them in the `DecimationOp` block.

 The CPU part of the pipeline performs the following tasks using independent blocks or operators:
- `FEngineCaptureOp`: Receive packets, sort and store them in a time-major layout (data dimensions: `time, chan, ant, pol, real_imag`).
- `DecimationOp`: Decimate the channels if need be (or average them on a GPU).
- `SaveImageOp`: Post-process the image for further storage.

On RTX 2080 Ti, the first iteration of the code can generate sky-images of size 64x64 (~1.8 deg resolution) with up to 90 channels (covering 2.25 MHz) for a single polarization without any significant packet loss. Alternatively, it has also been shown to produce images of size 96x96 (~1.4 deg resolution) for 22 channels (covering 3.3 MHz) for a single polarization. Although the current code can successfully produce sky images, the primary beam is undersampled, which limits the imaging frequencies to less than ~45 MHz. Furthermore, the images only cover a subset of the full bandwidth and only with a single polarization, which may reduce the detectability of any radio transients.

The main intent of this work is to be able to produce sky-images covering the full bandwidth and both the polarizations. Below I will describe the profiling results and metrics of the current GPU code, and I will discuss the new optimizations that allow us to obtain full bandwidth and full polarization images.


## GPU Profiling
### Profiling preparation
I used [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) to measure the overall performance (or system-level profiling), and [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) to measure the kernel metrics. See [Pearson (2020)](https://www.carlpearson.net/pdf/20200416_nsight.pdf) for an overview on these tools. For both tools, I ran EPIC with the following options to profile the `MoFFCorrelatorOp` block with a total runtime of 50 s:
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
### Results
Figure 1 shows screenshot from the Nsight Systems UI depicting the sequence of operations that are carried out in the correlator block. The typical processing times for a 40 ms gulp are as follows. 
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

The profiling data is available in the `profiling/systems` folder of the [epic-stream-processor](https://github.com/epic-astronomy/epic-stream-processor) repo. See the [user guide](https://docs.nvidia.com/nsight-systems/pdf/UserGuide.pdf) for additional information. 
> Notice the ~4 ms gap between reduce kernel and `D2H` copy of the reduced image. I'm unsure what is causing this, but I presume it is the time required to acquire a lock on the output ring buffer, which is an input to the `SaveOp` block. Although it could be a gap introduced by the profiler, it is unlikely for the gap to consistently exist only between the _reduce_ and the _copy_ steps.

![Screenshot from 2022-11-23 13-28-46](https://user-images.githubusercontent.com/4162508/203640587-3e21ed17-5943-466a-a822-a0ad8e05fb1f.png)
**Figure 1**: Screen shot of Nsight Systems UI. The timeline is zoomed-in to show a single sequence of imaging operations.

Although the FFT kernel runs in a different stream than VGrid and XGrid, they do not overlap. Furthermore, the FFT and the XGrid kernels appear to be the hotspots in imaging. To see if there is room for any improvement, let's take a look at the profiling results from Nsight Compute. I used the same options provided above to run the EPIC imager. See [Subrahmaniam (2021)](https://ericdarve.github.io/cme213-spring-2021/Lecture%20Slides/CME213_2021_CUDA_Profiling.pdf) for details on how to use Nsight Systems and Compute to optimize the kernels. Below I will describe the main metrics for each kernel.

### VGrid
This kernel grids the antenna data. The default settings provide a support size of a single cell. Put another way, each antenna only contributes to the cell it falls in. Figure 2a shows that the kernel is bandwidth-bound and that memory is more heavily utilized than compute SMs. That means, for a given unit of memory bandwidth utilization, non-optimal work is being done.

![image](https://user-images.githubusercontent.com/4162508/203662798-ee8b27be-e1ab-4586-bd5b-39525d4d9330.png)
**Figure 2a:** Screen shot of GPU throughput for VGrid from the Nsight Compute UI. This kernel achieves ~12% of the max compute throughout while ~72% memory throughput, making it memory bound.
![image](https://user-images.githubusercontent.com/4162508/203663075-b25430e3-e505-438c-8099-3aac853e4dec.png)
**Figure 2b:** Screen shot of Performance vs. Arithmetic Intensity for VGrid from the Nsight Compute UI. The first and second square markers from bottom indicate the peak double and single precision performances (or _rooflines_), respectively. This kernel achieves ~12% of the max compute throughout while ~72% memory throughput, making it bandwidth bound.

The first roofline in Figure 2b indicates the peak double-precision performance, while the second one indicates the peak single-precision performance. The achieved performance is shown as a filled-green circle that is about half of the peak double-precision performance. Note that the kernel achieves nearly 100% warp occupancy. That means it keeps the GPU fully occupied during the computation, however, does not achieve a high FLOPS/byte. It is because the kernel spends most of the time accessing data from the global memory.
<!--
![image](https://user-images.githubusercontent.com/4162508/203663666-c18a9fcf-7a46-42fd-a146-a3f8ce247c2e.png)
>TODO: Improve global memory accesses in VGrid. However, note that this step only takes about 7% of the total process time, and hence should be prioritized below the rest.
-->

### vector_2d_fft
This kernel preforms inverse FFT on the gridded antenna data to produce sky images. The FFT kernel is also bandwidth bound similar to the VGrid kernel. The FLOPS/bytes (3.06) is higher than the expected double-precision roofline (0.57) suggesting the amount of work done per byte is near the optimal value.

![image](https://user-images.githubusercontent.com/4162508/203670528-5e20494f-bebd-4914-98f9-cab06de118e8.png)
**Figure 3a:** Same as Fig. 2a but for `vector_2d_fft`

![image](https://user-images.githubusercontent.com/4162508/203671353-75de93f1-c126-4dc8-b050-6e064653fb82.png)
**Figure 3b:** Same as Fig. 2b but for `vector_2d_fft`

The theoretical warp occupancy (fraction of the total available active warps occupied) is, however, only 50%. It is because the shared memory required by all the active warps is greater than the available memory that is restricting the warp occupancy to 50%.

<!--
![image](https://user-images.githubusercontent.com/4162508/203671257-a3942405-d9c0-405a-9711-c64e45b5d043.png)
![image](https://user-images.githubusercontent.com/4162508/203671871-60470b60-f028-473d-b5fd-700eb7328d60.png)

Furthermore, the above figure shows that the warp occupancy can be improved by increasing block size. 
-->

### XGrid
This kernel is responsible for producing the cross-multiplied products (XX, YY, XY*, YX*). In this case, only the XX product is generated. Although XGrid is bandwidth bound similar to VGrid and FFT, it has as >80% bandwidth usage as shown in Figure 4a, indicating efficient memory access patterns. However, its performance is still below the double precision roofline (see Fig. 4b).

![image](https://user-images.githubusercontent.com/4162508/203678892-2584babd-b108-4288-ab92-55ccd2460782.png)
**Figure 4a:** Same as Fig 2a but for `XGrid` 
![image](https://user-images.githubusercontent.com/4162508/203678979-6238ebe0-8601-4a4b-b59d-e3378405a12e.png)
**Figure 4b:** Same as Fig 2b but for `XGrid` 

## Discussion
The profiling results clearly indicate that all the kernels used in the correlator block are bandwidth bound. It is primarily due to each kernel making large stores and/or fetches from the global memory: 
1. VGrid makes large stores to the global memory. For instance, write-size for an 64x64 image with a gulp size of 1000 sequences, each with 90 channels and a single polarization is ~2 GiB. 
2. The FFT kernel although performs above the double precision roofline, it must fetch the gridded data and store the iFFTed data to the global memory.
3. The XGrid kernel must again fetch the iFFTed data from the global memory to derive cross multiplication products and write them back to the global memory.

Furthermore, for each accumulation, initializing the gridded and cross multiplied data regions consumes ~19% of the total accumulation time. These bottlenecks can be eliminated by coalescing these kernels and performing computations using on-chip memory (registers and shared memory). 

A new package called [cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/index.html) allows us to perform FFT operations completely using registers and shared memory. Let us consider the memory requirements for a 64x64 image. At complex double precision each image occupies 64 KiB. That means we will only be able to run a single block per SM and two with single precision. This leads to <50% warp occupancy thereby increasing the overall runtime. However, with half (16 bit) precision, we can fit two 64x64 images in a 32 KiB shared memory block. This permits us to image a single image with both polarizations within the same block. On RTX 2080 Ti, it takes about ~13 ms to perform iFFT on a 1000-sequence glup with 132 channels for both the polarizations, which is similar to the time taken to iFFT an identical gulp but with only 90 channels and a single polarization. Below I describe how the VGrid and XGrid kernels can be combined with cuFFTDx to perform gridding and cross multiplication using on-chip memory.

### Gridding
The previous [optimization](https://github.com/epic-astronomy/Memos/blob/master/PDFs/003_Romein_Optimization.pdf) to the gridding involved processing each antenna on a separate thread and storing the gridded data back to the shared memory. The code used a support size of a single grid cell (or a pill box). However, cuFFTDx has strict requirements for the block dimenions, which may not always equal the number of antennas. Hence, in the new code, the threads are divided into _tiles_ (or groups) of `supportxsupport` using the `cooperative_groups` functionality provided by the CUDA programming model. Each tile computes all the grid values of a single antenna and atomically adds to the grid in the shared memory. The Gridding Convolution Function (GCF) is a zero-order prolate spheroid, and is stored as a texture and the factors are fetched using appropriate grid coordinates. Because the distances between antennas and grid coordinates are most likely to take non-integral values, the values are fetched using normalized coordinates. This gridding mode avoids storing data back into the global memory and facilitates antenna-polarization dependent GCFs.

### FFT
The gridded data is transferred from shared memory into thread registers for computation by cuFFTDx. Because cuFFTDx only supports 1D FFTs, the iFFT is preformed row-wise and then column-wise with appropriate normalizations to avoid values oveflowing the half precision. 

### Cross Multiplications
With half precision, each register stores two complex values (in a RRII layout), which is exploited to store a pixel value for both polarizations in a single register. Due to this layout, the cross multiplication products (XX, YY, XY*, YX*) for each pixel can be derived within a single thread without requiring any global memory reads. To avoid atomic writes to the global memory, each channel is imaged in its own dedicated block. This will also eliminate the need to initialize the output memory block to 0 because the first write to the memory will be assignment and the rest will be in-place additions.

With these optimizations, on `NVIDIA A4500`, it takes ~30 ms to generate a 64x64 sky-image with 112 (2.8 MHz) channels and both polarizations although with a support size of only 2 cells. Moreover, the total memory requirement including constant and texture memory is less than 200 MiB. The bandwidth is limited because of the shared memory constraint of 32 KiB per block that allows only 112 simultaneous active blocks (2 blocks per SM) on the GPU. Any GPU with an SM count>66 (e.g., RTX 2080 Ti) would allow us to generate images with the full bandwidth of 3.3 MHz.


## Future Improvements
* The image size is limited to 64x64 due to shared memory constraints. Although a single 128x128 image can fit in the shared memory, it will restrict the maximum active blocks per SM to one. Because the FFT is performed row-wise followed by column-wise, the FFT can be split to multiple blocks where each block performs row-wise FFT first and acquires column-wise data from adjacent blocks.  This inter-block communication can be achieved in two ways:

    1. Map a global memory block into the L2 cache and use it as a workspace for inter-block memory transfers.
    2. Use clusters of blocks with distributed shared memory. This does not require any global memory accesss, however, requires GPUs with compute capability of at least 9.0.





