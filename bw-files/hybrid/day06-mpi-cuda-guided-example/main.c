// This is a simple MPI+CUDA program that does the following:
//
// 1. On the each host, fill an array with consecutive numbers, starting at 0
//    for the first host, incrementing up to some amount, and then incrementing
//    from there up across each host.
// 2. Copy the array from the each host to its device.
// 3. On the device, multiply each number in the array by 2.
// 4. Copy the array from the device to the host.
// 5. Gather the arrays from each host onto a single host.
// 6. On that host, print the result.
//
// Author: Aaron Weeden, Shodor, 2016

// Import library so we can use MPI functions
#include <mpi.h>

// Import library so we can call printf()
#include <stdio.h>

// Import library so we can call exit() and use EXIT_FAILURE, as well as call
// malloc() and free()
#include <stdlib.h>

// Import library with model parameters
#include "params.h"

// Import library with declarations for functions that use CUDA
#include "simple-cuda.h"

// Declare variable for the host array
int * HostNums;

// Declare functions that will be defined later
void TryMPI(int const err);
void TryMalloc(void * const err);

// Start the program
int main(int argc, char ** argv)
{
  // Initialize MPI, set rank and size
  MPI_Init(&argc, &argv);
  int rank;
  TryMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int size;
  TryMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  // Make it so we can catch the return code of MPI functions
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  // Allocate memory for the host array
  TryMalloc(HostNums = (int*)malloc(BYTE_COUNT));

  // Allocate memory for the device array
  AllocateDeviceMemory();

  // On the each host, fill an array with consecutive numbers, starting at 0
  // for the first host, incrementing up to some amount, and then incrementing
  // from there up across each host.
  for (int i = 0; i < NUM_COUNT; i++)
  {
    HostNums[i] = rank * NUM_COUNT + i;
  }

  // 1. Copy the array from each host to its device
  // 2. On the device, multiply each number in the array by 2
  // 3. Copy the array from the device to the host
  MultBy2OnDevice();

  // Gather the arrays from each host onto a single host.
  int allNums[size * NUM_COUNT];
  TryMPI(MPI_Gather(HostNums, // Where in memory is the message that will be 
                              // sent?
                    NUM_COUNT, // How many elements of the message will be sent
                               // from each process?
                    MPI_INT, // What is the datatype of the elements to be
                             // sent?
                    allNums, // Where in memory should the received message 
                             // be stored?
                    NUM_COUNT, // How many elements of the message will be sent
                               // from each process?
                    MPI_INT, // What is the datatype of the elements to be
                             // received?
                    0, // Who is the receiver?
                    MPI_COMM_WORLD)); // Which processes are involved in this 
                                      // communication?

  // On that host, print the result
  if (rank == 0)
  {
    printf("Result:\n");
    for (int i = 0; i < size * NUM_COUNT; i++)
    {
      printf("\tallNums[%d] = %d\n", i, allNums[i]);
    }
  }

  // De-allocate memory for the device array
  FreeDeviceMemory();

  // De-allocate memory for the host array
  free(HostNums);

  // Finish calling MPI functions
  TryMPI(MPI_Finalize());

  return 0;
}

// Checks if the return to an MPI function is MPI_SUCCESS, and exits if it is
// not
void TryMPI(int const err)
{
  if (err != MPI_SUCCESS)
  {
    char string[120];
    int resultlen;
    MPI_Error_string(err, string, &resultlen);
    fprintf(stderr, "ERROR: MPI: %s\n", string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }
}

// Define a function to check whether a malloc() call was successful
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}

