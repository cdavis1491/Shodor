/* Model of heat diffusion - a 2D rectangular environment has its left edge with
   a fixed non-zero heat; the other 3 edges have fixed zero heat. The middle of 
   the environment starts with zero heat, but as the model advances, the heat
   of each cell in the middle gets set to the average of its 4 neighbors. The
   model advances until the amount of overall heat change from one time step to
   the next is sufficiently small.
 */

/*******************************************************************************
  IMPORT LIBRARIES
 ******************************************************************************/
#include <stdio.h>
#include "params.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
// Define the number of CUDA threads in each CUDA warp (group of threads that
// execute instructions in lock-step)
#define THREADS_PER_WARP 32

// Define the maximum number of CUDA warps in each CUDA block
#define MAX_WARPS_PER_BLOCK 16

// Define the number of CUDA threads in each CUDA block
#define THREADS_PER_BLOCK ((THREADS_PER_WARP) * (MAX_WARPS_PER_BLOCK))

// Define the number of CUDA blocks in each CUDA grid
#define BLOCKS_PER_GRID 1

/*******************************************************************************
  DECLARE GLOBAL VARIABLES
 ******************************************************************************/
extern int CellCount; // Total number of cells in the environment
extern int CellCountWithoutEdges; // Total number of cells in the environment,
                                  // not counting the edges
extern int CellFloatByteCount; // Total number of bytes if there are enough
                               // floats for each cell
extern int CellFloatByteCountWithoutEdges; // Total number of bytes if there are
                                           // enough floats for each cell, not
                                           // counting the edges
extern int CellCharByteCount; // Total number of bytes if there are enough chars
                              // for each cell
extern float * HostHeats; // Array of heat values for each cell (host memory)
float * DeviceHeats; // Array of heat values for each cell (device memory)
extern float * HostNewHeats; // Array of heat values for each cell in the next
                             // time step (host memory)
float * DeviceNewHeats; // Array of heat values for each cell in the next time
extern float * HostDiffs; // Array of differences between the heat values for
                          // each cell in the current and next time steps (host
                          // memory)
float * DeviceDiffs; // Array of differences between the heat values for each
extern char * OutputStr; // String to output at each time step
extern bool IsStillRunning; // Used to keep track of whether the model should
                            // continue into the next time step
extern int TimeIdx; // The current time step

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void TryCuda(cudaError_t const err);
__global__ void AverageNeighborHeats(float const * const DeviceHeats,
                                     float * const DeviceNewHeats,
                                     float * const DeviceDiffs,
                                     int const CellCountWithoutEdges);
__global__ void AdvanceHeats(float * const DeviceHeats,
                             float const * const DeviceNewHeats,
                             int const CellCountWithoutEdges);

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
// Define a function to check whether a CUDA call was successful
void TryCuda(cudaError_t const err)
{
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

extern "C"
{
  void InitDeviceMemory()
  {
    TryCuda(cudaMalloc((void**)&DeviceHeats, CellFloatByteCount));
    TryCuda(cudaMalloc((void**)&DeviceNewHeats,
                       CellFloatByteCountWithoutEdges));
    TryCuda(cudaMalloc((void**)&DeviceDiffs, CellFloatByteCountWithoutEdges));
  }
}

extern "C"
{
  void AdvanceHeatsOnDevice()
  {
    cudaMemcpy(DeviceNewHeats, HostNewHeats, CellFloatByteCountWithoutEdges,
               cudaMemcpyHostToDevice);

    AdvanceHeats<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(DeviceHeats,
                                                         DeviceNewHeats,
                                                         CellCountWithoutEdges);

    cudaMemcpy(HostHeats, DeviceHeats, CellFloatByteCount,
               cudaMemcpyDeviceToHost);
  }
}

extern "C"
{
  void AverageNeighborHeatsOnDevice()
  {
    TryCuda(cudaMemcpy(DeviceHeats, HostHeats, CellFloatByteCount,
                       cudaMemcpyHostToDevice));
  
    AverageNeighborHeats<<<BLOCKS_PER_GRID,
                           THREADS_PER_BLOCK>>>(DeviceHeats, DeviceNewHeats,
                                                DeviceDiffs,
                                                CellCountWithoutEdges);
  
    TryCuda(cudaMemcpy(HostNewHeats, DeviceNewHeats,
                       CellFloatByteCountWithoutEdges, cudaMemcpyDeviceToHost));
    TryCuda(cudaMemcpy(HostDiffs, DeviceDiffs,
                       CellFloatByteCountWithoutEdges, cudaMemcpyDeviceToHost));
  }
}

// Preconditions: Heats has not been updated at TimeIdx and cellIdx
// Postconditions: NewHeats has been updated at TimeIdx and cellIdx
//                 Diffs has been updated at TimeIdx and cellIdx
__global__ void AverageNeighborHeats(float const * const DeviceHeats,
                                     float * const DeviceNewHeats,
                                     float * const DeviceDiffs,
                                     int const CellCountWithoutEdges)
{
  int const cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cellIdx < CellCountWithoutEdges)
  {
    DeviceNewHeats[cellIdx] = 0.25 * (DeviceHeats[NEW_TO_OLD(cellIdx) -
                                                  COLUMN_COUNT] +
                                      DeviceHeats[NEW_TO_OLD(cellIdx) - 1] +
                                      DeviceHeats[NEW_TO_OLD(cellIdx) + 1] +
                                      DeviceHeats[NEW_TO_OLD(cellIdx) +
                                                  COLUMN_COUNT]);

    DeviceDiffs[cellIdx] = DeviceNewHeats[cellIdx] -
                           DeviceHeats[NEW_TO_OLD(cellIdx)];
  }
}

// Preconditions: NewHeats has been updated at TimeIdx
// Postconditions: Heats has been updated at TimeIdx
__global__ void AdvanceHeats(float * const DeviceHeats,
                             float const * const DeviceNewHeats,
                             int const CellCountWithoutEdges)
{
  int const cellIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cellIdx < CellCountWithoutEdges)
  {
    DeviceHeats[NEW_TO_OLD(cellIdx)] = DeviceNewHeats[cellIdx];
  }
}

extern "C"
{
  void FinalizeDeviceMemory()
  {
    // Free device memory
    TryCuda(cudaFree(DeviceDiffs));
    TryCuda(cudaFree(DeviceNewHeats));
    TryCuda(cudaFree(DeviceHeats));
  }
}

