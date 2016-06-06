#include <omp.h>
#include <sched.h>

#ifdef DEBUG
  // Prints information about OpenMP threads executing the given function
  void DebugFunction(char const * name)
  {
    printf("DEBUG: Thread %d (%d total) on cpu %d calling "
           "%s at %f\n", omp_get_thread_num(),
           omp_get_num_threads(), sched_getcpu(), name, omp_get_wtime());
  }
  
  // Prints number of threads executing a loop in the given function
  void DebugLoop(int const i, char const * name)
  {
    if (i == 0)
    {
      printf("DEBUG: %s loop has %d threads\n", name, omp_get_num_threads());
    }
  }
#endif

// Checks if the return to malloc() is NULL, and exits if it is
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}

// Checks if the return to sprintf() is negative, and exits if it is
void TrySprintf(int const err)
{
  if (err < 0)
  {
    fprintf(stderr, "ERROR in sprintf\n");
    exit(EXIT_FAILURE);
  }
}

// Checks if the return to memcpy() is NULL, and exits if it is
void TryMemcpy(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "ERROR in memcpy\n");
    exit(EXIT_FAILURE);
  }
}

