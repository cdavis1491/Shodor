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
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "params.h"
#include "fire-cuda.h"

/*******************************************************************************
  DECLARE MACROS
 ******************************************************************************/
#define OMP_NUM_THREADS 16

/*******************************************************************************
  DECLARE GLOBAL VARIABLES
 ******************************************************************************/
int RandomSeed; // Seed for the random number generator
int TreeCount; // Total number of trees in the forest
int TreeIntByteCount; // Total number of bytes if there are enough integers
                      // for each tree
bool IsStillBurning; // True iff there is at least 1 tree still burning
int CurrentTime; // The current time step
int * BurnStates; // Array with one element for each tree
int * NewBurnStates; // Array with one element for each tree
int ** NeighborIndices; // Array with one element for each tree; each element is
                        // and array with an element for the index of each
                        // neighbor in the BurnStates/NewBurnStates arrays
int * NeighborIndexCounts; // length of NeighborIndices array
bool IsStillBurning; // True iff there is at least 1 tree still burning
int * RandomPercents; // Array with one element for each pair of non-burning
                      // and burning trees
int * RandomPercentIndices; // Array with the same number of elements as the
                            // RandomPercents array. Each element is an index
                            // into the BurnStates/NewBurnStates array that
                            // contains a non-burning tree next to a burning
                            // tree
int RandomPercentCount; // length of RandomPercents array

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void InitRandomSeed();
void InitTreeCount();
void InitCPUMemory();
  void TryMalloc(void * const err);
void InitNeighborIndices();
void OutputData(int const timeIdx);
void BurnNewTrees();
  void GetRandomPercentIndices();
  void CatchFire();
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

/*  // Debug
  int threadCounter = 0;
  #pragma omp parallel reduction(+:threadCounter)
  {
    #pragma omp parallel reduction(+:threadCounter)
    {
      threadCounter++;
    }
  }
  printf("threadCounter: %d\n", threadCounter);*/

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

  #pragma omp parallel sections num_threads(3)
  {
    #pragma omp section
    {
      InitCPUMemory();
    }
    #pragma omp section
    {
      InitGPUMemory();
    }
    #pragma omp section
    {
      InitGPUGridSize();
    }
  }

  #pragma omp parallel sections num_threads(3)
  {
    #pragma omp section
    {
      InitNeighborIndices();
    }
    #pragma omp section
    {
      InitBurnStates();
    }
    #pragma omp section
    {
      InitRandomStates();
    }
  }

  IsStillBurning = true;
  for (int timeIdx = 0; IsStillBurning; timeIdx++)
  {
    #pragma omp parallel sections num_threads(2)
    {
      #pragma omp section
      {
        // Print the forest
        OutputData(timeIdx);
      }

      #pragma omp section
      {
        // Find trees that are not on fire yet and try to catch them on fire
        // from burning neighbor trees
        BurnNewTrees();
      }
    }

    // Copy new burn states into current burn states
    AdvanceStates();
  }

  FinalizeMemory();

  return 0;
}

void InitRandomSeed()
{
  #ifdef RANDOM_SEED
    RandomSeed = RANDOM_SEED;
  #else
    RandomSeed = time(NULL);
  #endif
}

void InitTreeCount()
{
  TreeCount = ROW_COUNT * COLUMN_COUNT;
  TreeIntByteCount = TreeCount * sizeof(int);
}

// Create dynamic memory for arrays
void InitCPUMemory()
{
  // Allocate space for the burn state arrays
  TryMalloc(BurnStates    = (int*)malloc(TreeIntByteCount));
  TryMalloc(NewBurnStates = (int*)malloc(TreeIntByteCount));

  // Allocate space for the random percents and random percent indices
  TryMalloc(RandomPercents       = (int*)malloc(TreeIntByteCount));
  TryMalloc(RandomPercentIndices = (int*)malloc(TreeIntByteCount));

  // Allocate space for the neighbor indices and neighbor index counts
  TryMalloc(NeighborIndices = (int**)malloc(TreeCount * sizeof(int*)));
  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
    TryMalloc(NeighborIndices[treeIdx] = (int*)malloc(MAX_NEIGHBOR_COUNT *
                                                                  sizeof(int)));
  }
  TryMalloc(NeighborIndexCounts = (int*)malloc(TreeIntByteCount));
}

// Checks if the return to malloc() is NULL, and exits if it is
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}

// Have each tree point to its neighbors
void InitNeighborIndices()
{
  // Spawn threads to complete the loop iterations in parallel
  #pragma omp parallel for num_threads(OMP_NUM_THREADS-2)
  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
/*      // Debug
    printf("I am thread %d\n", omp_get_thread_num()); */

    // Initialize the number of neighbors
    NeighborIndexCounts[treeIdx] = 0;

    // Point to the left neighbor, if not on the left edge
    if (treeIdx % COLUMN_COUNT > 0)
    {
      NeighborIndices[treeIdx][NeighborIndexCounts[treeIdx]++] = treeIdx - 1;
    }

    // Point to the top neighbor, if not on the top edge
    if (treeIdx / COLUMN_COUNT > 0)
    {
      NeighborIndices[treeIdx][NeighborIndexCounts[treeIdx]++] = treeIdx -
                                                                 COLUMN_COUNT;
    }

    // Point to the right neighbor, if not on the right edge
    if (treeIdx % COLUMN_COUNT < COLUMN_COUNT-1)
    {
      NeighborIndices[treeIdx][NeighborIndexCounts[treeIdx]++] = treeIdx + 1;
    }

    // Point to the bottom neighbor, if not on the bottom edge
    if (treeIdx / COLUMN_COUNT < ROW_COUNT-1)
    {
      NeighborIndices[treeIdx][NeighborIndexCounts[treeIdx]++] = treeIdx +
                                                                 COLUMN_COUNT;
    }
  }

  // Debug
/*  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
    printf("Tree %d: %d neighbors at ", treeIdx, NeighborIndexCounts[treeIdx]);
    for (int i = 0; i < NeighborIndexCounts[treeIdx]; i++)
    {
      printf("%d ", NeighborIndices[treeIdx][i]);
    }
    printf("\n");
  }
*/
}

// Print the forest
void OutputData(int const timeIdx)
{
  printf("Time step %d\n", timeIdx);
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    for (int columnIdx = 0; columnIdx < COLUMN_COUNT; columnIdx++)
    {
      printf("%d ", BurnStates[rowIdx * COLUMN_COUNT + columnIdx]);
    }
    printf("\n");
  }
  printf("\n");
}

// Find trees that are not on fire yet and try to catch them on fire from
// burning neighbor trees
void BurnNewTrees()
{
  // Make list of random percent indices
  GetRandomPercentIndices();

  // Make list of random percents
  GetRandomPercents();

  // Set new burn states of trees using the random percents
  CatchFire();
}

// Make list of random percent indices
void GetRandomPercentIndices()
{
  RandomPercentCount = 0;
  IsStillBurning = false; // Assume there are no trees burning any more

  // Loop through all trees in parallel
  #pragma omp parallel for num_threads(OMP_NUM_THREADS-1)
  for (int treeIdx = 0; treeIdx < TreeCount; treeIdx++)
  {
    // If a tree is burning
    if (BurnStates[treeIdx] == BURNING)
    {
      IsStillBurning = true;

      // Set the tree's next state to be burnt out
      NewBurnStates[treeIdx] = BURNT_OUT;

      // Look at all the tree's neighbors
      for (int i = 0; i < NeighborIndexCounts[treeIdx]; i++)
      {
        int const neighborIdx = NeighborIndices[treeIdx][i];

        // If the neighbor is not burning
        if (BurnStates[neighborIdx] == NOT_BURNING)
        {
          // One thread at a time so there is no race condition
          #pragma omp critical
          {
            // Add the neighbor's index to the list to correspond with a
            // random percent that will be calculated later
            RandomPercentIndices[RandomPercentCount++] = neighborIdx;
          }
        }
      }
    }
  }

/*  // Debug
  printf("Random percent indices\n");
  for (int randomPercentIdx = 0; randomPercentIdx < RandomPercentCount;
       randomPercentIdx++)
  {
    printf("%d: %d\n", randomPercentIdx,
           RandomPercentIndices[randomPercentIdx]);
  }*/
}

// Set new burn states of trees using the random percents
void CatchFire()
{
  // Loop through all trees in parallel
  #pragma omp parallel for num_threads(OMP_NUM_THREADS-1)
  for (int percentIdx = 0; percentIdx < RandomPercentCount; percentIdx++)
  {
    if (RandomPercents[percentIdx] < BURN_CHANCE)
    {
      NewBurnStates[RandomPercentIndices[percentIdx]] = BURNING;
    }
  }

/*  // Debug
  printf("New burn states\n");
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    for (int columnIdx = 0; columnIdx < COLUMN_COUNT; columnIdx++)
    {
      printf("%d ", NewBurnStates[rowIdx * COLUMN_COUNT + columnIdx]);
    }
    printf("\n");
  }
  printf("\n");*/
}

void FinalizeMemory()
{
  // Free dynamic memory on the GPU
  FinalizeGPUMemory();

  free(NeighborIndexCounts);
  for (int treeIdx = TreeCount - 1; treeIdx >= 0; treeIdx--)
  {
    free(NeighborIndices[treeIdx]);
  }
  free(NeighborIndices);
  free(RandomPercentIndices);
  free(RandomPercents);
  free(NewBurnStates);
  free(BurnStates);
}

