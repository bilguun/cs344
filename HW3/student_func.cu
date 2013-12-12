#include <cmath>
#include <stdio.h>

#include "reference_calc.cpp"
#include "utils.h"

typedef float (*my_func) ( const float&,const float&);

__device__ float min_op(const float & x, const float& y)
{
    return x<y?x:y;
}

__device__ float max_op(const float & x, const float& y)
{
    return x>y?x:y;
}

template<my_func func>
__global__ void my_min_max(float* const d_logLum,
                                  const size_t max_size)
{
    int gId = blockDim.x * blockIdx.x + threadIdx.x;
    int tId = threadIdx.x;

    if (gId >= max_size)
        return;

    extern __shared__ float sh_mem[];

    sh_mem[tId] = d_logLum[gId];

    __syncthreads();

    for(int i = blockDim.x/2; i > 0 ; i>>=1)
    {   
        if(tId < i)
        {
           
            sh_mem[tId] = func(sh_mem[tId],sh_mem[i +tId]);
        }   
        __syncthreads();
    }
    if (0 == tId)
    {
       d_logLum[0] = func(d_logLum[0], sh_mem[0]);
    }
}

__global__
void histogram(const float * const d_logLuminance, unsigned int *d_hist, float minv, const float range, const int numBins)
{
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        float item = d_logLuminance[tid];
        int bin = (item - minv) / range * numBins;
        atomicAdd(&(d_hist[bin]), 1);
}
__global__
void exscan(unsigned int * d_hist, unsigned int * const d_cdf, const int numBins)
{
        extern __shared__ unsigned int tmp[];
        int tid = threadIdx.x;
        tmp[tid] = (tid>0) ? d_hist[tid-1] : 0;
        __syncthreads();
        for(int offset = 1; offset < numBins; offset *= 2)
        {
                unsigned int lv = tmp[tid];
                __syncthreads();
                if(tid + offset < numBins)
                {
                        tmp[tid + offset] += lv;
                }
                __syncthreads();
        }
        d_cdf[tid] = tmp[tid];
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    
    float * d_min, * d_max;
    const int size = numRows*numCols*sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&d_min,size));

   // Copy array to two arrays one for max and one for min


    checkCudaErrors(cudaMemcpy(d_min,d_logLuminance,size,cudaMemcpyDeviceToDevice));
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    const int threads = 1024;
    const int blocks = numRows * numCols / threads;
    
    //Calling Kernel for first time to reduce threads.
    
    //For min
    my_min_max<min_op><<<blocks,threads,threads*sizeof(float)>>>
        (d_min,numRows*numCols);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
    checkCudaErrors(cudaMemcpy(&min_logLum,d_min,sizeof(float),cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(d_min));

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //For max
    checkCudaErrors(cudaMalloc((void**)&d_max,size));

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(d_max,d_logLuminance,size,cudaMemcpyDeviceToDevice));

    my_min_max<max_op><<<blocks,threads,threads*sizeof(float)>>>
        (d_max,numRows*numCols);
   
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  
    checkCudaErrors(cudaMemcpy(&max_logLum,d_max,sizeof(float),cudaMemcpyDeviceToHost));
 
    checkCudaErrors(cudaFree(d_max));

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
    //Range  
    float range =  max_logLum - min_logLum;

    //Histogram
    unsigned int* d_hist, *h_hist;
    cudaMalloc((void**)&d_hist, sizeof(unsigned int) * numBins);
    cudaMemset(d_hist, 0, sizeof(unsigned int)*numBins);
    histogram<<<blocks, threads>>>(d_logLuminance, d_hist, min_logLum, range, numBins);  
    
    //CDF
    exscan<<<1, 1024, sizeof(unsigned int) * 1024>>>(d_hist, d_cdf, numBins);
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
}
