/* Udacity Homework 3
   HDR Tone-mapping
  Background HDR
  ==============
  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.
  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.
  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.
  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.
  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.
  Background Chrominance-Luminance
  ================================
  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.
  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.
  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.
  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.

  Tone-mapping
  ============
  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.
  Example
  -------
  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9
  histo with 3 bins: [4 7 3]
  cdf : [4 11 14]
  Your task is to calculate this cumulative distribution by following these
  steps.
*/

#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"

__global__
void histogram_kernel(unsigned int* d_bins, const float* d_in, const int bin_count, const float lum_min, const float lum_max, const int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= size) {
        return;
    }

    float lum_range = lum_max - lum_min;
    int bin = ((d_in[index] - lum_min) / lum_range) * bin_count;

    atomicAdd(&d_bins[bin], 1);
}

__global__
void scan_kernel(unsigned int* d_bins, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= size) {
        return;
    }

    for(int i = 1; i <= size; i *= 2) {
        int spot = index - i;

        unsigned int val = 0;
        if(spot >= 0) {
            val = d_bins[spot];
        }
        __syncthreads();
        if(spot >= 0) {
            d_bins[index] += val;
        }
        __syncthreads();
    }
}
// calculate reduce max or min and stick the value in d_answer.
__global__
void reduceMinMax_kernel(const float* const d_in, float* d_out, const size_t size, int minmax) {
    extern __shared__ float shared[];

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int threadId = threadIdx.x;

    // we have 1 thread per block, so copying the entire block should work fine
    if(index < size) {
        shared[threadId] = d_in[index];
    } else {
        if(minmax == 0) {
            shared[threadId] = FLT_MAX;
        } else {
            shared[threadId] = -FLT_MAX;
        }
    }

    // wait for all threads to copy the memory
    __syncthreads();

    // don't do any thing with memory if we happen to be far off ( I don't know how this works with
    // sync threads so I moved it after that point )
    if(index >= size) {
        if(threadId == 0) {
            if(minmax == 0) {
                d_out[blockIdx.x] = FLT_MAX;
            } else {
                d_out[blockIdx.x] = -FLT_MAX;
            }
        }
        return;
    }

    for(unsigned int i = blockDim.x/2; i > 0; i /= 2) {
        if(threadId < i) {
            if(minmax == 0) {
                shared[threadId] = min(shared[threadId], shared[threadId + i]);
            } else {
                shared[threadId] = max(shared[threadId], shared[threadId + i]);
            }
        }

        __syncthreads();
    }

    if(threadId == 0) {
        d_out[blockIdx.x] = shared[0];
    }
}

int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) + 1;
}

float reduceMinMax(const float* const d_in, const size_t size, int minMax) {
    int BLOCK_SIZE = 32;
    // we need to keep reducing until we get to the amount that we consider
    // having the entire thing fit into one block size
    size_t curr_size = size;
    float* d_curr_in;

    checkCudaErrors(cudaMalloc(&d_curr_in, sizeof(float) * size));
    checkCudaErrors(cudaMemcpy(d_curr_in, d_in, sizeof(float) * size, cudaMemcpyDeviceToDevice));

    float* d_curr_out;
    dim3 thread_dim(BLOCK_SIZE);
    const int shared_mem_size = sizeof(float)*BLOCK_SIZE;

    while(true) {
        checkCudaErrors(cudaMalloc(&d_curr_out, sizeof(float) * get_max_size(curr_size, BLOCK_SIZE)));

        dim3 block_dim(get_max_size(size, BLOCK_SIZE));
        reduceMinMax_kernel<<<block_dim, thread_dim, shared_mem_size>>>
            (d_curr_in, d_curr_out, curr_size, minMax);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());


        // move the current input to the output, and clear the last input if necessary
        checkCudaErrors(cudaFree(d_curr_in));
        d_curr_in = d_curr_out;

        if(curr_size <  BLOCK_SIZE)
            break;

        curr_size = get_max_size(curr_size, BLOCK_SIZE);
    }

    // theoretically we should be
    float h_out;
    cudaMemcpy(&h_out, d_curr_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_curr_out);
    return h_out;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    const size_t size = numRows*numCols;
    min_logLum = reduceMinMax(d_logLuminance, size, 0);
    max_logLum = reduceMinMax(d_logLuminance, size, 1);

    printf("Got min value: %f\n", min_logLum);
    printf("Got max value: %f\n", max_logLum);
    printf("Number of Bins: %d\n\n", numBins);

    unsigned int* d_bins;
    size_t histo_size = sizeof(unsigned int)*numBins;

    //determine the size of the bins
    checkCudaErrors(cudaMalloc(&d_bins, histo_size));
    checkCudaErrors(cudaMemset(d_bins, 0, histo_size));

    //set up the thread dimensions
    dim3 thread_dim(1024);
    dim3 hist_block_dim(get_max_size(size, thread_dim.x));

    // deploy the kernel and calculate the image
    histogram_kernel<<<hist_block_dim, thread_dim>>>
        (d_bins, d_logLuminance, numBins, min_logLum, max_logLum, size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // get the output from the device
    unsigned int h_out[100];
    cudaMemcpy(&h_out, d_bins, sizeof(unsigned int) * 100, cudaMemcpyDeviceToHost);

    // set up the can block
    dim3 scan_block_dim(get_max_size(numBins, thread_dim.x));

    // release the kernels
    scan_kernel<<<scan_block_dim, thread_dim>>>(d_bins, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // copy the data back over
    cudaMemcpy(&h_out, d_bins, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_cdf, d_bins, histo_size, cudaMemcpyDeviceToDevice);
    checkCudaErrors(cudaFree(d_bins));

  //TODO
  /*Here are the steps you need to implement
    X - 1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    X - 2) subtract them to find the range
    X - 3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    X - 4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
}
