// This is a CUDA program that does the following:
//
// 1. On the host, fill the A and B arrays with random numbers
// 2. On the host, print the initial values of the A and B arrays
// 3. Copy the A and B arrays from the host to the device
// 4. On the device, add the A and B vectors and store the result in C
// 5. Copy the C array from the device to the host
// 6. On the host, print the result
//
// Author: Aaron Weeden, Shodor, 2016

// Import library so we can call printf()
#include <stdio.h>

// Import library so we can call exit(), malloc(), free(), random(), etc.
#include <stdlib.h>

// Import library so we can call time()
#include <time.h>

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
__global__ void AddVectors(int * const deviceA, int * const deviceB,
                           int * const deviceC, int const count);

// Start the program
int main()
{
  // Declare variables for the host and device arrays
  int * hostA;
  int * hostB;
  int * hostC;
  int * deviceA;
  int * deviceB;
  int * deviceC;

  // Allocate memory for the host arrays
  TryMalloc(hostA = (int*)malloc(BYTE_COUNT));
  TryMalloc(hostB = (int*)malloc(BYTE_COUNT));
  TryMalloc(hostC = (int*)malloc(BYTE_COUNT));

  // Allocate memory for the device arrays
  TryCuda(cudaMalloc((void**)&deviceA, BYTE_COUNT));
  TryCuda(cudaMalloc((void**)&deviceB, BYTE_COUNT));
  TryCuda(cudaMalloc((void**)&deviceC, BYTE_COUNT));

  // Initialize the random number generator
  srandom(time(NULL));

  // On the host, fill the A and B arrays with random numbers
  printf("Expected Result:\n");
  for (int i = 0; i < NUM_COUNT; i++)
  {
    hostA[i] = 100 * random() / RAND_MAX;
    hostB[i] = 100 * random() / RAND_MAX;
    printf("\thostC[%d] should be %d + %d\n", i, hostA[i], hostB[i]);
  }

  // Copy the A and B arrays from the host to the device
  TryCuda(cudaMemcpy(deviceA, hostA, BYTE_COUNT, cudaMemcpyHostToDevice));
  TryCuda(cudaMemcpy(deviceB, hostB, BYTE_COUNT, cudaMemcpyHostToDevice));

  // On the device, add the A and B vectors and store the result in C
  AddVectors<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(deviceA, deviceB, deviceC,
                                                     NUM_COUNT);

  // Copy the C array from the device to the host
  TryCuda(cudaMemcpy(hostC, deviceC, BYTE_COUNT, cudaMemcpyDeviceToHost));

  // On the host, print the result
  printf("Result:\n");
  for (int i = 0; i < NUM_COUNT; i++)
  {
    printf("\thostC[%d] = %d\n", i, hostC[i]);
  }

  // De-allocate memory for the device arrays
  TryCuda(cudaFree(deviceC));
  TryCuda(cudaFree(deviceB));
  TryCuda(cudaFree(deviceA));

  // De-allocate memory for the host arrays
  free(hostC);
  free(hostB);
  free(hostA);

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
__global__ void AddVectors(int * const deviceA, int * const deviceB,
                           int * const deviceC, int const count)
{
  // Calculate the unique ID for the current CUDA thread
  int const threadId = blockIdx.x * blockDim.x + threadIdx.x;

  // All threads whose thread ID is >= count will NOT do the following, thus
  // avoiding writing into un-allocated space.
  if (threadId < count)
  {
    deviceC[threadId] = deviceA[threadId] + deviceB[threadId];
  }
}

