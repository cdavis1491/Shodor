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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
#define ROW_COUNT 21
#define COLUMN_COUNT 25
#define INIT_LEFT_HEAT 100.0 // The left edge of the environment has this
                             // fixed heat
#define MIN_HEAT_DIFF 10 // If the overall system heat changes by less
                         // than this amount between two time steps, the
                         // model stops
#define OUTPUT_HEAT_LEN 6 // Number of characters needed to print each heat
                          // value
#define OUTPUT_DIGS_AFTER_DEC_PT 2  // Number of digits to print after the
                                    // decimal point for each heat value

// Convert index within NewHeats array (which only includes the middle cells)
// into index within Heats array (which also includes the edge cells)
#define NEW_TO_OLD(idx) ((idx) + (COLUMN_COUNT) + 1 + \
                         2 * ((idx) / ((COLUMN_COUNT)-2)))

// Convert index within Heats array into index within OutputStr string
#define OUTPUT_IDX(idx) ((idx) * ((OUTPUT_HEAT_LEN)+1) + \
                         (idx) / (COLUMN_COUNT))

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
int CellCount; // Total number of cells in the environment
int CellCountWithoutEdges; // Total number of cells in the environment, not
                           // counting the edges
int CellFloatByteCount; // Total number of bytes if there are enough floats for
                        // each cell
int CellFloatByteCountWithoutEdges; // Total number of bytes if there are
                                    // enough floats for each cell, not
                                    // counting the edges
int CellCharByteCount; // Total number of bytes if there are enough chars for
                       // each cell
float * HostHeats; // Array of heat values for each cell (host memory)
float * DeviceHeats; // Array of heat values for each cell (device memory)
float * HostNewHeats; // Array of heat values for each cell in the next time
                      // step (host memory)
float * DeviceNewHeats; // Array of heat values for each cell in the next time
                        // step (device memory)
float * HostDiffs; // Array of differences between the heat values for each
                   // cell in the current and next time steps (host memory)
float * DeviceDiffs; // Array of differences between the heat values for each
                     // cell in the current and next time steps (device memory)
char * OutputStr; // String to output at each time step
bool IsStillRunning; // Used to keep track of whether the model should continue
                     // into the next time step
int TimeIdx; // The current time step

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void TryCuda(cudaError_t const err);
void Init();
  void InitCellCounts();
  void InitMemory();
  void InitHeats();
    void SetOutputStr(int const cellIdx);
  void InitLeft();
  void InitOutputStrNewlines();
void Simulate();
  void SetNewHeats();
    __global__ void AverageNeighborHeats(float const * const DeviceHeats,
                                         float * const DeviceNewHeats,
                                         float * const DeviceDiffs,
                                         int const CellCountWithoutEdges);
  void PrintOutputStr();
  __global__ void AdvanceHeats(float * const DeviceHeats,
                               float const * const DeviceNewHeats,
                               int const CellCountWithoutEdges);
void Finalize();

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
int main()
{
  Init();
  Simulate();
  Finalize();
  return 0;
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

// Preconditions: none
// Postconditions: Heats has been initialized
//                 OutputStr has been initialized
void Init()
{
  InitCellCounts();
  InitMemory();
  InitHeats();
  InitLeft();
  InitOutputStrNewlines();
}

// Preconditions: none
// Postconditions: CellCount has been defined
//                 CellCountWithoutEdges has been defined
//                 CellFloatByteCount has been defined
//                 CellFloatByteCountWithoutEdges has been defined
//                 CellCharByteCount has been defined
void InitCellCounts()
{
  CellCount = ROW_COUNT * COLUMN_COUNT;
  CellCountWithoutEdges = CellCount - 2 * (ROW_COUNT + COLUMN_COUNT - 2);
  CellFloatByteCount = CellCount * sizeof(float);
  CellFloatByteCountWithoutEdges = CellCountWithoutEdges * sizeof(float);
  CellCharByteCount = CellCount * sizeof(char);
}

// Preconditions: CellFloatByteCount has been defined
//                CellFloatByteCountWithoutEdges has been defined
//                CellCharByteCount has been defined
// Postconditions: Memory has been allocated for Heats
//                 Memory has been allocated for NewHeats
//                 Memory has been allocated for Diffs
//                 Memory has been allocated for OutputStr
void InitMemory()
{
  // Allocate CPU memory
  TryMalloc(HostHeats    = (float*)malloc(CellFloatByteCount));
  TryMalloc(HostNewHeats = (float*)malloc(CellFloatByteCountWithoutEdges));
  TryMalloc(HostDiffs    = (float*)malloc(CellFloatByteCountWithoutEdges));
  TryMalloc(OutputStr    = (char*) malloc(CellCharByteCount *
                                          (OUTPUT_HEAT_LEN+1) +
                                          ROW_COUNT * sizeof(char)));

  // Allocate GPU memory
  TryCuda(cudaMalloc((void**)&DeviceHeats, CellFloatByteCount));
  TryCuda(cudaMalloc((void**)&DeviceNewHeats, CellFloatByteCountWithoutEdges));
  TryCuda(cudaMalloc((void**)&DeviceDiffs, CellFloatByteCountWithoutEdges));
}

// Preconditons: Memory has been allocated for Heats
// Postconditions: Heats has been initialized, except for the left edge
void InitHeats()
{
  for (int cellIdx = 0; cellIdx < CellCount; cellIdx++)
  {
    HostHeats[cellIdx] = 0.0;
    SetOutputStr(cellIdx);
  }
}

// Preconditons: Heats has been initialized, except for the left edge
// Postconditions: Heats has been initialized
void InitLeft()
{
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    int const cellIdx = rowIdx * COLUMN_COUNT;
    HostHeats[cellIdx] = INIT_LEFT_HEAT;
    SetOutputStr(cellIdx);
  }
}

// Preconditons: Memory has been allocated for OutputStr
// Postconditions: OutputStr has been initialized
void InitOutputStrNewlines()
{
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    OutputStr[OUTPUT_IDX((rowIdx+1) * (COLUMN_COUNT)) - 1] = '\n';
  }
}

// Preconditions: Heats has not been updated at TimeIdx and cellIdx
// Postconditions: OutputStr has been updated at TimeIdx and cellIdx
void SetOutputStr(int const cellIdx)
{
  char tmp[OUTPUT_HEAT_LEN+2];
  TrySprintf(sprintf(tmp,
                     "%*.*f ",
                     OUTPUT_HEAT_LEN,
                     OUTPUT_DIGS_AFTER_DEC_PT,
                     HostHeats[cellIdx]));
  TryMemcpy(memcpy(&(OutputStr[OUTPUT_IDX(cellIdx)]),
                   tmp,
                   OUTPUT_HEAT_LEN+1));
}

// Preconditons: Heats has been initialized
//               OutputStr has been initialized
// Postconditions: The simulation has run
void Simulate()
{
  IsStillRunning = true;
  for (TimeIdx = 0; IsStillRunning; TimeIdx++)
  {
    SetNewHeats();
    PrintOutputStr();

    cudaMemcpy(DeviceNewHeats, HostNewHeats, CellFloatByteCountWithoutEdges,
               cudaMemcpyHostToDevice);

    AdvanceHeats<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(DeviceHeats,
                                                         DeviceNewHeats,
                                                         CellCountWithoutEdges);

    cudaMemcpy(HostHeats, DeviceHeats, CellFloatByteCount,
               cudaMemcpyDeviceToHost);
  }
}

// Preconditions: Heats has not been updated at TimeIdx
// Postconditions: OutputStr has been updated at TimeIdx
//                 NewHeats has been updated at TimeIdx
//                 IsStillRunning has possibly been updated
void SetNewHeats()
{
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    SetOutputStr(NEW_TO_OLD(cellIdx));
  }

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

  // Prepare to stop the simulation if the heat is not changing much
  float totalDiff = 0.0;
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    totalDiff += HostDiffs[cellIdx];
  }
  if (totalDiff < MIN_HEAT_DIFF)
  {
    IsStillRunning = false;
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

// Preconditions: OutputStr has been updated at TimeIdx
// Postconditions: OutputStr has been printed at TimeIdx
void PrintOutputStr()
{
  printf("Time step %d\n%s\n", TimeIdx, OutputStr);
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

// Preconditions: The simulation has run
// Postconditions: Memory has been freed for Heats
//                 Memory has been freed for NewHeats
//                 Memory has been freed for Diffs
//                 Memory has been freed for OutputStr
void Finalize()
{
  // Free device memory
  TryCuda(cudaFree(DeviceDiffs));
  TryCuda(cudaFree(DeviceNewHeats));
  TryCuda(cudaFree(DeviceHeats));

  // Free host memory
  free(OutputStr);
  free(HostDiffs);
  free(HostNewHeats);
  free(HostHeats);
}

