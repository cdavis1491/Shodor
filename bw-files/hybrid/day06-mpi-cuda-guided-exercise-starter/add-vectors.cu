// This is an MPI+CUDA program that does the following:
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

/*
//christina commenting this out
// Import library so we can call time()
#include <time.h>
*/

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

// Declare variables for the host and device arrays
// the host variables are defined in main.c 
// since they are EXTERNALLy defined we will use extern here
extern int * HostA;
extern int * HostB;
extern int * HostC;

//we deleted these from main.c so they are only defined here
int * DeviceA;
int * DeviceB;
int * DeviceC;

// Declare functions that will be defined later
// try malloc is defined in other file
//void TryMalloc(void * const err);
void TryCuda(cudaError_t const err);
__global__ void AddVectors(int * const DeviceA, int * const DeviceB,
                           int * const DeviceC, int const count);

// Start the program

//now that we've commented out everything that has to do with host
//and removed our main we have floating lines that need to be in a function
//we need to group these inside an extern C so main.c can access this code
extern "C"
{

//we alreday have a main.c so we want to take out everything 
//except the CUDA parts. 

/*
//christina commented this out for reason above

int main(int argc, char ** argv)
{
  // Allocate memory for the host arrays
  TryMalloc(HostA = (int*)malloc(BYTE_COUNT));
  TryMalloc(HostB = (int*)malloc(BYTE_COUNT));
  TryMalloc(HostC = (int*)malloc(BYTE_COUNT));
*/

  //making leftover crap a function! 
  void InitDeviceMemory()
  {
  // Allocate memory for the device arrays
    TryCuda(cudaMalloc((void**)&DeviceA, BYTE_COUNT));
    TryCuda(cudaMalloc((void**)&DeviceB, BYTE_COUNT));
    TryCuda(cudaMalloc((void**)&DeviceC, BYTE_COUNT));
  }
/* 
//this is taken care of by the host now
  // Initialize the random number generator
  srandom(time(NULL));
*/

/*
  //also done on the host so we don't need this

  // On the host, fill the A and B arrays with random numbers
  printf("Expected Result:\n");
  for (int i = 0; i < NUM_COUNT; i++)
  {
    HostA[i] = 100 * random() / RAND_MAX;
    HostB[i] = 100 * random() / RAND_MAX;
    printf("\tHostC[%d] should be %d + %d\n", i, HostA[i], HostB[i]);
  }
*/

  //making leftover crap a function! 
  void AddVectorsOnDevice()
  {
    // Copy the A and B arrays from the host to the device
    TryCuda(cudaMemcpy(DeviceA, HostA, BYTE_COUNT, cudaMemcpyHostToDevice));
    TryCuda(cudaMemcpy(DeviceB, HostB, BYTE_COUNT, cudaMemcpyHostToDevice));

    // On the device, add the A and B vectors and store the result in C
    AddVectors<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(DeviceA, DeviceB, DeviceC,
                                                     NUM_COUNT);

    // Copy the C array from the device to the host
    TryCuda(cudaMemcpy(HostC, DeviceC, BYTE_COUNT, cudaMemcpyDeviceToHost));
  }

/* 
   //dont care about stuff on hose
  // On the host, print the result
  printf("Result:\n");
  for (int i = 0; i < NUM_COUNT; i++)
  {
    printf("\tHostC[%d] = %d\n", i, HostC[i]);
  }
*/

//making leftover crap a function! 

  void FinalizeDeviceMemory()
  {
    // De-allocate memory for the device arrays
    TryCuda(cudaFree(DeviceC));
    TryCuda(cudaFree(DeviceB));
    TryCuda(cudaFree(DeviceA));
  }
/*
  //has to do with host. 
  // De-allocate memory for the host arrays
  free(HostC);
  free(HostB);
  free(HostA);

  return 0;
}
*/ 

} //close extern C 

/*
//host only funtion so we don't need it

// Define a function to check whether a malloc() call was successful
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}
*/

//keep the 2 CUDA functions
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
__global__ void AddVectors(int * const DeviceA, int * const DeviceB,
                           int * const DeviceC, int const count)
{
  // Calculate the unique ID for the current CUDA thread
  int const threadId = blockIdx.x * blockDim.x + threadIdx.x;

  // All threads whose thread ID is >= count will NOT do the following, thus
  // avoiding writing into un-allocated space.
  if (threadId < count)
  {
    DeviceC[threadId] = DeviceA[threadId] + DeviceB[threadId];
  }
}

