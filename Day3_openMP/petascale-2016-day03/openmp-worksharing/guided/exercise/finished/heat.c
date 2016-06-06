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

// Define macro for number of threads
#define OMP_NUM_THREADS 32

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
float * Heats; // Array of heat values for each cell
float * NewHeats; // Array of heat values for each cell in the next time step
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
    void AverageNeighborHeats(int const cellIdx);
    float GetDiff(int const cellIdx);
  void PrintOutputStr();
  void AdvanceHeats();
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

  #ifdef DEBUG
    int threadCounter = 0;
    #pragma omp parallel reduction(+:threadCounter)
    {
      #pragma omp parallel reduction(+:threadCounter)
      {
        threadCounter++;
      }
    }
    printf("DEBUG: threadCounter: %d\n", threadCounter);
  #endif

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
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

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
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

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
//                 Memory has been allocated for OutputStr
void InitMemory()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  TryMalloc(Heats     = (float*)malloc(CellFloatByteCount));
  TryMalloc(NewHeats  = (float*)malloc(CellFloatByteCountWithoutEdges));
  TryMalloc(OutputStr = (char*) malloc(CellCharByteCount *
                                       (OUTPUT_HEAT_LEN+1) +
                                       ROW_COUNT * sizeof(char)));
}

// Preconditons: Memory has been allocated for Heats
// Postconditions: Heats has been initialized, except for the left edge
void InitHeats()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
  for (int cellIdx = 0; cellIdx < CellCount; cellIdx++)
  {
    #ifdef DEBUG
      DebugLoop(cellIdx, __FUNCTION__);
    #endif

    Heats[cellIdx] = 0.0;
    SetOutputStr(cellIdx);
  }
}

// Preconditons: Heats has been initialized, except for the left edge
// Postconditions: Heats has been initialized
void InitLeft()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    #ifdef DEBUG
      DebugLoop(rowIdx, __FUNCTION__);
    #endif

    int const cellIdx = rowIdx * COLUMN_COUNT;
    Heats[cellIdx] = INIT_LEFT_HEAT;
    SetOutputStr(cellIdx);
  }
}

// Preconditons: Memory has been allocated for OutputStr
// Postconditions: OutputStr has been initialized
void InitOutputStrNewlines()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    #ifdef DEBUG
      DebugLoop(rowIdx, __FUNCTION__);
    #endif

    OutputStr[OUTPUT_IDX((rowIdx+1) * (COLUMN_COUNT)) - 1] = '\n';
  }
}

// Preconditions: Heats has not been updated at TimeIdx and cellIdx
// Postconditions: OutputStr has been updated at TimeIdx and cellIdx
void SetOutputStr(int const cellIdx)
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  char tmp[OUTPUT_HEAT_LEN+2];
  TrySprintf(sprintf(tmp,
                     "%*.*f ",
                     OUTPUT_HEAT_LEN,
                     OUTPUT_DIGS_AFTER_DEC_PT,
                     Heats[cellIdx]));
  TryMemcpy(memcpy(&(OutputStr[OUTPUT_IDX(cellIdx)]),
                   tmp,
                   OUTPUT_HEAT_LEN+1));
}

// Preconditons: Heats has been initialized
//               OutputStr has been initialized
// Postconditions: The simulation has run
void Simulate()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  IsStillRunning = true;
  for (TimeIdx = 0; IsStillRunning; TimeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(TimeIdx, __FUNCTION__);
    #endif

    SetNewHeats();

    #pragma omp parallel sections num_threads(2)
    {
      #pragma omp section
      {
        PrintOutputStr();
      }
      #pragma omp section
      {
        AdvanceHeats();
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
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  float totalDiff = 0.0; // Keep track of how much system heat has changed
  #pragma omp parallel for reduction(+:totalDiff)
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    #ifdef DEBUG
      DebugLoop(cellIdx, __FUNCTION__);
    #endif

    #pragma omp parallel sections num_threads(2)
    {
      #pragma omp section
      {
        SetOutputStr(NEW_TO_OLD(cellIdx));
      }
      #pragma omp section
      {
        AverageNeighborHeats(cellIdx);
        totalDiff += GetDiff(cellIdx);
      }
    }
  }
  // Prepare to stop the simulation if the heat is not changing much
  if (totalDiff < MIN_HEAT_DIFF)
  {
    IsStillRunning = false;
  }
}

// Preconditions: Heats has not been updated at TimeIdx and cellIdx
// Postconditions: NewHeats has been updated at TimeIdx and cellIdx
void AverageNeighborHeats(int const cellIdx)
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  NewHeats[cellIdx] = (Heats[NEW_TO_OLD(cellIdx) - COLUMN_COUNT] +
                       Heats[NEW_TO_OLD(cellIdx) - 1] +
                       Heats[NEW_TO_OLD(cellIdx) + 1] +
                       Heats[NEW_TO_OLD(cellIdx) + COLUMN_COUNT]) * 0.25;
}

// Preconditions: Heats has not been updated at TimeIdx and cellIdx
//                NewHeats has been updated at TimeIdx and cellIdx
// Postconditions: none
float GetDiff(int const cellIdx)
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  return NewHeats[cellIdx] - Heats[NEW_TO_OLD(cellIdx)];
}

// Preconditions: OutputStr has been updated at TimeIdx
// Postconditions: OutputStr has been printed at TimeIdx
void PrintOutputStr()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  printf("Time step %d\n%s\n", TimeIdx, OutputStr);
}

// Preconditions: NewHeats has been updated at TimeIdx
// Postconditions: Heats has been updated at TimeIdx
void AdvanceHeats()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS-1)
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    #ifdef DEBUG
      DebugLoop(cellIdx, __FUNCTION__);
    #endif

    Heats[NEW_TO_OLD(cellIdx)] = NewHeats[cellIdx];
  }
}

// Preconditions: The simulation has run
// Postconditions: Memory has been freed for Heats
//                 Memory has been freed for NewHeats
//                 Memory has been freed for OutputStr
void Finalize()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  free(OutputStr);
  free(NewHeats);
  free(Heats);
}

