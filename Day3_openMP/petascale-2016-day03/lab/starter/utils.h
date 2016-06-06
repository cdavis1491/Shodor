/*******************************************************************************
  IMPORT LIBRARIES
 ******************************************************************************/
#include <omp.h>
#include <sched.h>
#include <stdlib.h>

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
// Define a function that returns a random integer less than a given maximum
int GetRandom(int const max) {
  return max * ((double)random() / RAND_MAX);
}

#ifdef DEBUG
  // Prints information about OpenMP threads executing the given function
  void DebugFunction(char const * name)
  {
    printf("DEBUG: Thread %d (%d total) on cpu %d calling "
           "%s at %f\n", omp_get_thread_num(),
           omp_get_num_threads(), sched_getcpu(), name, omp_get_wtime());
  }

  // Prints information about OpenMP threads executing the given function
  //  that is in a loop
  void DebugFunctionInLoop(int const i, char const * name)
  {
    if (i == 0)
    {
      DebugFunction(name);
    }
  }

  // Prints number of threads executing a loop in the given function
  void DebugLoop(int const i, char const * name)
  {
    if (i == 0)
    {
      printf("DEBUG: %s loop has %d threads\n", name, omp_get_num_threads());
    }
  }

  // Prints number of threads executing a nested loop in the given function
  void DebugNestedLoop(int const i, int const j, char const * name)
  {
    if (j == 0)
    {
      DebugLoop(i, name);
    }
  }

  // Prints number of threads executing a nested loop in the given function
  void DebugDoublyNestedLoop(int const i, int const j, int const k,
                             char const * name)
  {
    if (k == 0)
    {
      DebugNestedLoop(i, j, name);
    }
  }
#endif

