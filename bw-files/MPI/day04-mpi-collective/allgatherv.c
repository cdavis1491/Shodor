#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.h"

int main(int argc, char ** argv)
{
  // Initialize MPI environment
  MPI_Init(&argc, &argv);

  // Make it so we can catch the return code of MPI functions
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  // Get rank and size
  int rank;
  TryMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int size;
  TryMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  // Declare the variables
  int nums[12];
  int recvcounts[4] = {2, 1, 5, 4};
  int displs[4] = {0, 2, 3, 8};
  int myNums[recvcounts[rank]];

  // All processes initialize the myNums variable
  // Seed the random number generator, include rank to try to be unique across
  // processes
  srandom(time(NULL) ^ rank);

  // Fill the array with random numbers less than 100
  for (int i = 0; i < recvcounts[rank]; i++)
  {
    myNums[i] = 100 * random() / RAND_MAX;
  }

  // Print the myNums array before the gather
  char myNumsStr[recvcounts[rank]*(2+1)];
  strcpy(myNumsStr, "");
  for (int i = 0; i < recvcounts[rank]; i++)
  {
    char numStr[3];
    sprintf(numStr, "%2d ", myNums[i]);
    strcat(myNumsStr, numStr);
  }
  printf("(1) before: rank %d: myNums: %s\n", rank, myNumsStr);

  // Gather
  TryMPI(MPI_Allgatherv(myNums, // Where in memory is the message that will be 
                                // sent?
                        recvcounts[rank], // How many elements of the message
                                          // will the current process send?
                        MPI_INT, // What is the datatype of the elements to be
                                 // sent?
                        nums, // Where in memory should the received message 
                              // be stored?
                        recvcounts, // How many elements of the message 
                                    // will be received per process?
                        displs, // What is the displacement of each process?
                        MPI_INT, // What is the datatype of the elements to be
                                 // received?
                        MPI_COMM_WORLD)); // Which processes are involved in
                                          // this communication?

  // Print the nums array after the gather
  char numsStr[12*(2+1)]; // 12 numbers, 2 digits + 1 space character each
  strcpy(numsStr, "");
  for (int i = 0; i < 12; i++)
  {
    char numStr[3];
    sprintf(numStr, "%2d ", nums[i]);
    strcat(numsStr, numStr);
  }
  printf("(2) after: rank %d: nums: %s\n", rank, numsStr);

  // Finish up with MPI
  TryMPI(MPI_Finalize());

  return 0;
}

