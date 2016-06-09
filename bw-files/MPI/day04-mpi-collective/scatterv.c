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
  int myNums[12];
  int recvcounts[4] = {2, 1, 5, 4};
  int displs[4] = {0, 2, 3, 8};

  // Rank 0 initializes the nums variable
  if (rank == 0)
  {
    // Seed the random number generator
    srandom(time(NULL));

    // Fill the array with random numbers less than 100
    for (int i = 0; i < 12; i++)
    {
      nums[i] = 100 * random() / RAND_MAX;
    }

    // Print the nums array
    printf("(1) before: nums: ");
    for (int i = 0; i < 12; i++)
    {
      printf("%d ", nums[i]);
    }
    printf("\n");
  }

  // Scatter
  TryMPI(MPI_Scatterv(nums, // Where in memory is the message that will be 
                            // sent?
                      recvcounts, // How many elements of the message will be
                                  // sent to each process?
                      displs, // What is the displacement of each process?
                      MPI_INT, // What is the datatype of the elements to be
                               // sent?
                      myNums, // Where in memory should the received message 
                              // be stored?
                      recvcounts[rank], // How many elements of the message will
                                        // be received by the current process?
                      MPI_INT, // What is the datatype of the elements to be
                               // received?
                      0, // Who is the sender?
                      MPI_COMM_WORLD)); // Which processes are involved in this 
                                        // communication?

  // Print the myNums array after the scatter
  char myNumsStr[recvcounts[rank]*(2+1)];
  strcpy(myNumsStr, "");
  for (int i = 0; i < recvcounts[rank]; i++)
  {
    char numStr[3];
    sprintf(numStr, "%2d ", myNums[i]);
    strcat(myNumsStr, numStr);
  }
  printf("(2) after: rank %d: myNums: %s\n", rank, myNumsStr);

  // Finish up with MPI
  TryMPI(MPI_Finalize());

  return 0;
}

