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
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "params.h"
#include "heat-cuda.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
#define OMP_NUM_THREADS 16

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
float * HostNewHeats; // Array of heat values for each cell in the next time
                      // step (host memory)
float * HostDiffs; // Array of differences between the heat values for each
                   // cell in the current and next time steps (host memory)
char * OutputStr; // String to output at each time step
bool IsStillRunning; // Used to keep track of whether the model should continue
                     // into the next time step
int TimeIdx; // The current time step

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void Init();
  void InitCellCounts();
  void InitMemory();
  void InitHeats();
    void SetOutputStr(int const cellIdx);
  void InitLeft();
  void InitOutputStrNewlines();
void Simulate();
  void SetNewHeats();
  void PrintOutputStr();
void Finalize();

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
int main()
{
  // Set the number of OpenMP threads
  omp_set_num_threads(OMP_NUM_THREADS);

  // Enable nested OpenMP regions
  omp_set_nested(true);

  Init();
  Simulate();
  Finalize();
  return 0;
}

// Preconditions: none
// Postconditions: Heats has been initialized
//                 OutputStr has been initialized
void Init()
{
  InitCellCounts();
  InitMemory();
  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section
    {
      InitHeats();
      InitLeft();
    }
    #pragma omp section
    {
      InitOutputStrNewlines();
    }
  }
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
  InitDeviceMemory();
}

// Preconditons: Memory has been allocated for Heats
// Postconditions: Heats has been initialized, except for the left edge
void InitHeats()
{
  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
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
  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
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
  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
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
    #pragma omp parallel sections num_threads(2)
    {
      #pragma omp section
      {
        PrintOutputStr();
      }
      #pragma omp section
      {
        AdvanceHeatsOnDevice();
      }
    }
  }
}

// Preconditions: Heats has not been updated at TimeIdx
// Postconditions: OutputStr has been updated at TimeIdx
//                 NewHeats has been updated at TimeIdx
//                 IsStillRunning has possibly been updated
void SetNewHeats()
{
  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section
    {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS-1)
      for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
      {
        SetOutputStr(NEW_TO_OLD(cellIdx));
      }
    }
    #pragma omp section
    {
      AverageNeighborHeatsOnDevice();
    }
  }

  // Prepare to stop the simulation if the heat is not changing much
  float totalDiff = 0.0;
  #pragma omp parallel for reduction(+:totalDiff)
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    totalDiff += HostDiffs[cellIdx];
  }
  if (totalDiff < MIN_HEAT_DIFF)
  {
    IsStillRunning = false;
  }
}

// Preconditions: OutputStr has been updated at TimeIdx
// Postconditions: OutputStr has been printed at TimeIdx
void PrintOutputStr()
{
  printf("Time step %d\n%s\n", TimeIdx, OutputStr);
}

// Preconditions: The simulation has run
// Postconditions: Memory has been freed for Heats
//                 Memory has been freed for NewHeats
//                 Memory has been freed for Diffs
//                 Memory has been freed for OutputStr
void Finalize()
{
  // Free device memory
  FinalizeDeviceMemory();

  // Free host memory
  free(OutputStr);
  free(HostDiffs);
  free(HostNewHeats);
  free(HostHeats);
}

