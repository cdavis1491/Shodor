/*******************************************************************************
  IMPORT LIBRARIES
 ******************************************************************************/
#include <curand.h>
#include <curand_kernel.h>
#include <sched.h>
#include <stdio.h>
#include "params.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
#define THREADS_PER_WARP 32
#define MAX_WARPS_PER_BLOCK 16
// Get a unique thread ID within a CUDA grid
#define GET_THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)

/*******************************************************************************
  DECLARE GLOBAL VARIABLES
 ******************************************************************************/
extern int TreeCount;
extern int TreeIntByteCount;
extern int * BurnStates;
extern int * NewBurnStates;
extern int * RandomPercents;
extern int RandomPercentCount;
extern int RandomSeed;
int BlockCount;
int ThreadsPerBlock;
int * D_BurnStates;
int * D_NewBurnStates;
curandState * D_RandomStates;
int * D_RandomPercents;

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void TryCuda(cudaError_t err);
__global__ void InitRandomStates_kernel(int const randomSeed,
                                        curandState * const states);
__global__ void InitBurnStates_kernel(int * const D_BurnStates,
                                      int * const D_NewBurnStates);
__global__ void GetRandomPercents_kernel(curandState * const states,
                                         int * const D_RandomPercents);
__global__ void AdvanceStates_kernel(int const * const D_NewBurnStates,
                                     int * const D_BurnStates);

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
extern "C"
{
  // Calculate the number of CUDA blocks per grid and threads per block
  void InitGPUGridSize()
  {
    ThreadsPerBlock = THREADS_PER_WARP * MAX_WARPS_PER_BLOCK;
    BlockCount = ceil((double)TreeCount / ThreadsPerBlock);

/*    // Debug
    printf("Set grids to have %d blocks and %d threads\n",
            BlockCount, ThreadsPerBlock);*/
  }

  // Allocate dynamic memory on the GPU
  void InitGPUMemory()
  {
    TryCuda(cudaMalloc((void**)&D_BurnStates,     TreeIntByteCount));
    TryCuda(cudaMalloc((void**)&D_NewBurnStates,  TreeIntByteCount));
    TryCuda(cudaMalloc((void**)&D_RandomStates,   TreeCount *
                                                          sizeof(curandState)));
    TryCuda(cudaMalloc((void**)&D_RandomPercents, TreeIntByteCount));
  }

  // Initialize the random states array for the GPU
  void InitRandomStates()
  {
    InitRandomStates_kernel<<<BlockCount, ThreadsPerBlock>>>(RandomSeed,
                                                             D_RandomStates);
  }

  // Initialize the burn states arrays on the GPU
  void InitBurnStates()
  {
/*    // Debug
    for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
    {
      printf("Before Tree %d: %d %d\n", treeIdx, BurnStates[treeIdx],
             NewBurnStates[treeIdx]);
    }
*/

    // Launch a GPU kernel
    InitBurnStates_kernel<<<BlockCount, ThreadsPerBlock>>>(D_BurnStates,
                                                           D_NewBurnStates);

    // Copy the burn states from the GPU
    TryCuda(cudaMemcpy(BurnStates, // destination pointer
                       D_BurnStates, // source pointer
                       TreeIntByteCount, // # of bytes
                       cudaMemcpyDeviceToHost)); // direction

    TryCuda(cudaMemcpy(NewBurnStates, // destination pointer
                       D_NewBurnStates, // source pointer
                       TreeIntByteCount, // # of bytes
                       cudaMemcpyDeviceToHost)); // direction

    // Light the middle tree on fire
    int const middleIdx = 0.5 * COLUMN_COUNT * ROW_COUNT;
    BurnStates[middleIdx] = BURNING;

    // Copy the burn states back to the GPU
    TryCuda(cudaMemcpy(D_BurnStates, // destination pointer
                       BurnStates, // source pointer
                       TreeIntByteCount, // # of bytes
                       cudaMemcpyHostToDevice)); // direction
  }

  // Make list of random percents
  void GetRandomPercents()
  {
    GetRandomPercents_kernel<<<BlockCount, ThreadsPerBlock>>>(D_RandomStates,
                                                              D_RandomPercents);

    TryCuda(cudaMemcpy(RandomPercents, // destination pointer
                       D_RandomPercents, // source pointer
                       TreeIntByteCount, // # of bytes
                       cudaMemcpyDeviceToHost)); // direction

/*    // Debug
    printf("Random percents\n");
    for (int randomPercentIdx = 0; randomPercentIdx < RandomPercentCount;
         randomPercentIdx++)
    {
      printf("%d: %d\n", randomPercentIdx, RandomPercents[randomPercentIdx]);
    }*/
  }

  // Copy new burn states into current burn states
  void AdvanceStates()
  {
    TryCuda(cudaMemcpy(D_NewBurnStates, // destination pointer
                       NewBurnStates, // source pointer
                       TreeIntByteCount, // # of bytes
                       cudaMemcpyHostToDevice)); // direction

    // Copy new burn states into current burn states on the GPU
    AdvanceStates_kernel<<<BlockCount, ThreadsPerBlock>>>(D_NewBurnStates,
                                                          D_BurnStates);

    TryCuda(cudaMemcpy(BurnStates, // destination pointer
                       D_BurnStates, // source pointer
                       TreeIntByteCount, // # of bytes
                       cudaMemcpyDeviceToHost)); // direction
  }

  // Free dynamic memory on the GPU
  void FinalizeGPUMemory()
  {
    TryCuda(cudaFree(D_RandomPercents));
    TryCuda(cudaFree(D_BurnStates));
    TryCuda(cudaFree(D_NewBurnStates));
    TryCuda(cudaFree(D_RandomStates));
  }
}

// Check if the return value of a CUDA function is an error; if it is, print
// the error and exit
void TryCuda(cudaError_t err)
{
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Initialize the random states array for the GPU
__global__ void InitRandomStates_kernel(int const randomSeed,
                                        curandState * const states)
{
  int const threadId = GET_THREAD_ID;

  curand_init(randomSeed, // seed
              threadId, // sequence
              0, // offset
              &states[threadId]);
}

// Initialize the burn states arrays on the GPU
__global__ void InitBurnStates_kernel(int * const D_BurnStates,
                                      int * const D_NewBurnStates)
{
  int const threadId = GET_THREAD_ID;
  D_BurnStates[threadId] = D_NewBurnStates[threadId] = NOT_BURNING;
}

// Make list of random percents on the GPU
__global__ void GetRandomPercents_kernel(curandState * const states,
                                         int * const D_RandomPercents)
{
  int const threadId = GET_THREAD_ID;
  D_RandomPercents[threadId] = curand_uniform(&states[threadId]) * 100;
}

// Copy new burn states into current burn states on the GPU
__global__ void AdvanceStates_kernel(int const * const D_NewBurnStates,
                                     int * const D_BurnStates)
{
  int const threadId = GET_THREAD_ID;
  D_BurnStates[threadId] = D_NewBurnStates[threadId];
}

