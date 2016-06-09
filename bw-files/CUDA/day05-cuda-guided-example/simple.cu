// This is a simple CUDA program that does the following:
//
// 1. On the host, fill an array with consecutive numbers, starting at 0.
// 2. Copy the array from the host to the device.
// 3. On the device, multiply each number in the array by 2.
// 4. Copy the array from the device to the host.
// 5. On the host, print the result
//
// Author: Aaron Weeden, Shodor, 2016

// Import library so we can call printf()
#include <stdio.h>

// Import library so we can call exit() and use EXIT_FAILURE, as well as call
// malloc() and free()
#include <stdlib.h>

// Define the number of numbers in each array
#define NUM_COUNT 10

// Define the number of bytes in each array
#define BYTE_COUNT ((NUM_COUNT) * sizeof(int))

// Define the number of CUDA threads in each CUDA warp (group of threads that
// execute instructions in lock-step)
#define THREADS_PER_WARP 32

// Define the maximum number of CUDA warps in each CUDA block
#define MAX_WARPS_PER_BLOCK 16

// Define the number of CUDA threads in each CUDA block
#define THREADS_PER_BLOCK ((THREADS_PER_WARP) * (MAX_WARPS_PER_BLOCK))

// Define the number of CUDA blocks in each CUDA grid
#define BLOCKS_PER_GRID 1

// Declare functions that will be defined later
void TryMalloc(void * const err);
void TryCuda(cudaError_t const err);
__global__ void MultBy2(int * const deviceNums, int const count);

// Start the program
int main()
{
  // Declare variables for the host and device arrays
  int * hostNums;
  int * deviceNums;

  // Allocate memory for the host array
  TryMalloc(hostNums = (int*)malloc(BYTE_COUNT));

  // Allocate memory for the device
  TryCuda(cudaMalloc((void**)&deviceNums, BYTE_COUNT));

  // On the host, fill an array with consecutive numbers, starting at 0
  for (int i = 0; i < NUM_COUNT; i++)
  {
    hostNums[i] = i;
  }

  // Copy the array from the host to the device
  TryCuda(cudaMemcpy(deviceNums, hostNums, BYTE_COUNT, cudaMemcpyHostToDevice));

  // On the device, multiply each number in the array by 2
  MultBy2<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(deviceNums, NUM_COUNT);

  // Copy the array from the device to the host
  TryCuda(cudaMemcpy(hostNums, deviceNums, BYTE_COUNT, cudaMemcpyDeviceToHost));

  // On the host, print the result
  printf("Result:\n");
  for (int i = 0; i < NUM_COUNT; i++)
  {
    printf("\thostNums[%d] = %d\n", i, hostNums[i]);
  }

  // De-allocate memory for the device array
  TryCuda(cudaFree(deviceNums));

  // De-allocate memory for the host array
  free(hostNums);

  return 0;
}

// Define a function to check whether a malloc() call was successful
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
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
__global__ void MultBy2(int * const deviceNums, int const count)
{
  // Calculate the unique ID for the current CUDA thread
  int const threadId = blockIdx.x * blockDim.x + threadIdx.x;

  // All threads whose thread ID is >= count will NOT do the following, thus
  // avoiding writing into un-allocated space.
  if (threadId < count)
  {
    // The current thread indexes the device array using its thread ID and
    // multiplies that element by 2.
    deviceNums[threadId] *= 2;
  }
}

