/* Model of a forest fire - a 2D rectangular grid of trees is initialized
   with the middle tree caught on fire. At each time step, trees that are not
   on fire yet check their neighbors to the north, east, south, and west; for
   each of these burning trees, the non-burning tree catches fire with some
   percent chance. The model runs until the fire burns out. At the end of the
   simulation, the program outputs the total percentage of trees burned and the
   number of iterations the fire lasted.
 */

/* Author: Aaron Weeden, Shodor, 2016 */

/* Naming convention:
   ALL_CAPS for constants
   CamelCase for globals and functions
   lowerCase for locals
 */
/*******************************************************************************
  IMPORT LIBRARIES
 ******************************************************************************/
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
// Define macros for simulation parameters
#define MAX_NEIGHBOR_COUNT 4 // Each tree has at most this many neighbors
#define ROW_COUNT 21
#define COLUMN_COUNT 25
#define BURN_CHANCE 50

// Define macros for tree states
#define NOT_BURNING 0
#define BURNING 1
#define BURNT_OUT 2

// Define macro for number of threads
#define OMP_NUM_THREADS 32

/*******************************************************************************
  DECLARE GLOBAL VARIABLES
 ******************************************************************************/
int TreeCount; // Total number of trees in the forest
int TreeIntByteCount; // Total number of bytes if there are enough integers
                      // for each tree
int TreeCharByteCount; // Total number of bytes if there are enough characters
                       // for each tree
int TimeIdx; // The current time step
int * BurnStates; // Array with one element for each tree
int * NewBurnStates; // Array with one element for each tree
char * OutputStr; // String to output at each time step
bool IsStillBurning; // true iff there are any burning trees left

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void Init();
  void InitRandomSeed();
  void InitTreeCount();
  void InitMemory();
  void InitBurnStates();
  void InitOutputStrNewlines();
void Simulate();
  void SetNewStates();
    void TryBurn(int const idx);
  void PrintOutputStr();
  void AdvanceStates();
void Finalize();
  void PrintFinalCounts();
  void FinalizeMemory();

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
// Postconditions: The random number generator has been seeded
//                 BurnStates, NewBurnStates, and OutputStr have been
//                   initialized
void Init()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section
    {
      InitRandomSeed();
    }
    #pragma omp section
    {
      InitTreeCount();
    }
  }

  InitMemory();

  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section
    {
      InitBurnStates();
    }
    #pragma omp section
    {
      InitOutputStrNewlines();
    }
  }
}

// Preconditions: none
// Postconditions: The random number generator has been seeded
void InitRandomSeed()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  int randomSeed;
  #ifdef RANDOM_SEED
    randomSeed = RANDOM_SEED;
  #else
    randomSeed = time(NULL);
  #endif
  srandom(randomSeed);
}

// Preconditions: none
// Postconditions: TreeCount, TreeIntByteCount, and TreeCharByteCount have been
//                   defined
void InitTreeCount()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  TreeCount = ROW_COUNT * COLUMN_COUNT;
  TreeIntByteCount  = TreeCount * sizeof(int);
  TreeCharByteCount = TreeCount * sizeof(char);
}

// Preconditions: TreeIntByteCount and TreeCharByteCount have been defined
// Postconditions: Memory has been allocated for BurnStates, NewBurnStates,
//                 and OutputStr
void InitMemory()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  TryMalloc(BurnStates    = (int*) malloc(TreeIntByteCount));
  TryMalloc(NewBurnStates = (int*) malloc(TreeIntByteCount));
  TryMalloc(OutputStr     = (char*)malloc(TreeCharByteCount +
                                          ROW_COUNT * sizeof(char)));
}

// Preconditions: Memory has been allocated for BurnStates and NewBurnStates
// Postconditions: BurnStates and NewBurnStates have been initialized
void InitBurnStates()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(treeIdx, __FUNCTION__);
    #endif

    BurnStates[treeIdx] = NewBurnStates[treeIdx] = NOT_BURNING;
  }

  // Light the middle tree on fire
  int const middleIdx = 0.5 * COLUMN_COUNT * ROW_COUNT;
  BurnStates[middleIdx] = BURNING;
}

// Preconditions: Memory has been allocated for OutputStr
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

    OutputStr[(rowIdx + 1) * (COLUMN_COUNT + 1) - 1] = '\n';
  }
}

// Preconditions: The random number generator has been seeded
//                BurnStates, NewBurnStates, and OutputStr have been initialized
// Postconditions: The simulation has run
void Simulate()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  IsStillBurning = true;
  // Loop iteration start postconditions: BurnStates, NewBurnStates,
  //   OutputStr, and IsStillBurning have not been updated at TimeIdx
  // Loop iteration end preconditions: BurnStates, NewBurnStates, OutputStr, and
  //   IsStillBurning have been updated at TimeIdx
  for (TimeIdx = 0; IsStillBurning; TimeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(TimeIdx, __FUNCTION__);
    #endif

    SetNewStates();

    #pragma omp parallel sections num_threads(2)
    {
      #pragma omp section
      {
        PrintOutputStr();
      }
      #pragma omp section
      {
        AdvanceStates();
      }
    }
  }
}

// Preconditions: BurnStates has not been updated at TimeIdx
//                The random number generator has been seeded
// Postconditions: OutputStr has been updated at TimeIdx
//                 NewBurnStates has been updated at TimeIdx
//                 IsStillBurning has been updated at TimeIdx
void SetNewStates()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  IsStillBurning = false; // Assume there are no trees burning any more

  // Loop through all trees in parallel
  #pragma omp parallel for reduction(||:IsStillBurning)
  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(treeIdx, __FUNCTION__);
    #endif

    // Update OutputStr with this tree's data in this time step
    OutputStr[treeIdx + treeIdx / COLUMN_COUNT] = BurnStates[treeIdx] + '0';

    // If the tree is burning
    if (BurnStates[treeIdx] == BURNING)
    {
      IsStillBurning = true;

      // Set the tree's next state to be burnt out
      NewBurnStates[treeIdx] = BURNT_OUT;

      // Point to the left neighbor, if not on the left edge
      if (treeIdx % COLUMN_COUNT > 0)
      {
        TryBurn(treeIdx-1);
      }
  
      // Point to the top neighbor, if not on the top edge
      if (treeIdx / COLUMN_COUNT > 0)
      {
        TryBurn(treeIdx-COLUMN_COUNT);
      }
  
      // Point to the right neighbor, if not on the right edge
      if (treeIdx % COLUMN_COUNT < COLUMN_COUNT-1)
      {
        TryBurn(treeIdx+1);
      }
  
      // Point to the bottom neighbor, if not on the bottom edge
      if (treeIdx / COLUMN_COUNT < ROW_COUNT-1)
      {
        TryBurn(treeIdx+COLUMN_COUNT);
      }
    }
  }
}

// Preconditions: The random number generator has been seeded
//                BurnStates has not been updated at TimeIdx
// Postconditions: NewBurnStates for the given tree has been updated at TimeIdx
void TryBurn(int const idx)
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  // If the tree with the given index is not burning, and given some percent
  // chance, light the tree on fire
  if (BurnStates[idx] == NOT_BURNING && random() % 100 < BURN_CHANCE)
  {
    NewBurnStates[idx] = BURNING;
  }
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

// Preconditions: NewBurnStates has been updated at TimeIdx
// Postconditions: BurnStates has been updated at TimeIdx
void AdvanceStates()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS-1)
  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(treeIdx, __FUNCTION__);
    #endif

    BurnStates[treeIdx] = NewBurnStates[treeIdx];
  }
}

// Preconditions: The simulation has run
// Postconditions: The final burn % and time steps have been printed
//                 Memory has been freed for BurnStates, NewBurnStates, and
//                   OutputStr
void Finalize()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  PrintFinalCounts();
  FinalizeMemory();
}

// Preconditions: The simulation has run
//                Memory has not been freed for BurnStates
// Postconditions: The final burn % and time steps have been printed
void PrintFinalCounts()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  int count = 0;
  #pragma omp parallel for reduction(+:count)
  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(treeIdx, __FUNCTION__);
    #endif

    if (BurnStates[treeIdx] == BURNT_OUT)
    {
      count++;
    }
  }
  printf("%.2f percent of the forest burned in %d time steps\n",
         100.0 * count / TreeCount, TimeIdx-1);
}

// Preconditions: The simulation has run
// Postconditions: Memory has been freed for BurnStates, NewBurnStates, and
//                   OutputStr
void FinalizeMemory()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  free(OutputStr);
  free(NewBurnStates);
  free(BurnStates);
}

