//Udacity HW 4
//Radix Sorting

#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"


/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.
   Note: ascending order == smallest to largest
   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.
   Implementing Parallel Radix Sort with CUDA
   ==========================================
   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.
   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there
   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.
 */

__global__ void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned int* const d_input,
                      const int size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= size) {
        return;
    }
    unsigned int one = 1;
    int bin = ((d_input[index] & (one<<pass)) == (one<<pass)) ? 1 : 0;
    if(bin) {
      atomicAdd(&d_bins[1], 1);
    } else {
      atomicAdd(&d_bins[0], 1);
    }
}


__global__ void exclusive_scan_kernel(unsigned int pass,
                    unsigned int const * d_inputVals,
                    unsigned int * d_output,
                    const int size,
                    unsigned int base,
                    unsigned int threadSize) {
    int index = threadIdx.x + threadSize * base;
    unsigned int one = 1;

    if(index >= size)
        return;
      unsigned int val = 0;
    if(index > 0)  {
        val = ((d_inputVals[index-1] & (one<<pass))  == (one<<pass)) ? 1 : 0;
    } else {
        val = 0;
    }

    d_output[index] = val;

    __syncthreads();

    for(int s = 1; s <= threadSize; s *= 2) {
        int spot = index - s;

        if(spot >= 0 && spot >=  threadSize*base) {
          val = d_output[spot];
        }
        __syncthreads();
        if(spot >= 0 && spot >= threadSize*base) {
          d_output[index] += val;
        }
        __syncthreads();
    }
    if(base > 0) {
        d_output[index] += d_output[base*threadSize - 1];
    }

}

__global__ void move_kernel(
    unsigned int pass,
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* d_outputVals,
    unsigned int* d_outputPos,
    unsigned int* d_outputMove,
    unsigned int* const d_scanned,
    unsigned int  one_pos,
    const size_t numElems) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index >= numElems) {
        return;
    }

    unsigned int scan = 0;
    unsigned int base = 0;
    unsigned int one = 1;
    if( ( d_inputVals[index] & (one<<pass)) == (1<<pass)) {
        scan = d_scanned[index];
        base = one_pos;
    } else {
        scan = (index) - d_scanned[index];
        base = 0;
    }

    d_outputMove[index] = base+scan;
    d_outputPos[base+scan] = d_inputPos[index];
    d_outputVals[base+scan] = d_inputVals[index];

}

int get_max_size(int n, int d) {
    return (int)ceil((float) n / (float) d) + 1;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    unsigned int* d_bins;
    unsigned int  h_bins[2];
    unsigned int* d_scanned;
    unsigned int* d_moved;
    const size_t histo_size = 2*sizeof(unsigned int);
    const size_t arr_size = numElems*sizeof(unsigned int);

    checkCudaErrors(cudaMalloc(&d_bins, histo_size));
    checkCudaErrors(cudaMalloc(&d_scanned, arr_size));
    checkCudaErrors(cudaMalloc(&d_moved, arr_size));

    dim3 thread_dim(1024 );
    dim3 hist_block_dim(get_max_size(numElems, thread_dim.x));

    for(unsigned int pass = 0; pass < 32; pass++) {
        unsigned int one = 1;
        checkCudaErrors(cudaMemset(d_bins, 0, histo_size));
        checkCudaErrors(cudaMemset(d_scanned, 0, arr_size));
        checkCudaErrors(cudaMemset(d_outputVals, 0, arr_size));
        checkCudaErrors(cudaMemset(d_outputPos, 0, arr_size));

        histogram_kernel<<<hist_block_dim, thread_dim>>>(pass, d_bins, d_inputVals, numElems);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(&h_bins, d_bins, histo_size, cudaMemcpyDeviceToHost));

        for(int i = 0; i < get_max_size(numElems, thread_dim.x); i++) {
            exclusive_scan_kernel<<<dim3(1), thread_dim>>>
              (pass, d_inputVals, d_scanned, numElems, i, thread_dim.x);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
        }

        move_kernel<<<hist_block_dim, thread_dim>>>
          (pass, d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_moved, d_scanned, h_bins[0], numElems);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, arr_size, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, arr_size, cudaMemcpyDeviceToDevice));

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }

    checkCudaErrors(cudaFree(d_moved));
    checkCudaErrors(cudaFree(d_scanned));
    checkCudaErrors(cudaFree(d_bins));
}
