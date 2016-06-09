// This is a simple MPI+CUDA program that does the following:
//
// 1. On the each host, fill an array with consecutive numbers, starting at 0
//    for the first host, incrementing up to some amount, and then incrementing
//    from there up across each host.
// 2. Copy the array from the each host to its device.
// 3. On the device, multiply each number in the array by 2.
// 4. Copy the array from the device to the host.
// 5. Gather the arrays from each host onto a single host.
// 6. On that host, print the result.
//
// Author: Aaron Weeden, Shodor, 2016

// Import library so we can call printf()
#include <stdio.h>

// Import library so we can call exit() and use EXIT_FAILURE, as well as call
// malloc() and free()
#include <stdlib.h>

// Import library with model parameters
#include "params.h"

// Define the number of CUDA threads in each CUDA warp (group of threads that
// execute instructions in lock-step)
#define THREADS_PER_WARP 32

// Define the maximum number of CUDA warps in each CUDA block
#define MAX_WARPS_PER_BLOCK 16

// Define the number of CUDA threads in each CUDA block
#define THREADS_PER_BLOCK ((THREADS_PER_WARP) * (MAX_WARPS_PER_BLOCK))

// Define the number of CUDA blocks in each CUDA grid
#define BLOCKS_PER_GRID 1

// Declare variable for device memory
extern int * HostNums;
int * DeviceNums;

// Declare functions that will be defined later
void TryCuda(cudaError_t const err);
__global__ void MultBy2(int * const DeviceNums, int const count);

extern "C"
{
  void AllocateDeviceMemory()
  {
    // Allocate memory for the device array
    TryCuda(cudaMalloc((void**)&DeviceNums, BYTE_COUNT));
  }

  void MultBy2OnDevice()
  {
    // Copy the array from each host to its device
    TryCuda(cudaMemcpy(DeviceNums, HostNums, BYTE_COUNT,
                       cudaMemcpyHostToDevice));

    // On the device, multiply each number in the array by 2
    MultBy2<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(DeviceNums, NUM_COUNT);

    // Copy the array from the device to the host
    TryCuda(cudaMemcpy(HostNums, DeviceNums, BYTE_COUNT,
                       cudaMemcpyDeviceToHost));
  }

  void FreeDeviceMemory()
  {
    // De-allocate memory for the device array
    TryCuda(cudaFree(DeviceNums));
  }
}

// Define a function to check whether a CUDA call was successful
void TryCuda(cudaError_t const err)
{
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Define a function which will be executed on a CUDA device
__global__ void MultBy2(int * const DeviceNums, int const count)
{
  // Calculate the unique ID for the current CUDA thread
  int const threadId = blockIdx.x * blockDim.x + threadIdx.x;

  // All threads whose thread ID is >= count will NOT do the following, thus
  // avoiding writing into un-allocated space.
  if (threadId < count)
  {
    // The current thread indexes the device array using its thread ID and
    // multiplies that element by 2.
    DeviceNums[threadId] *= 2;
  }
}

