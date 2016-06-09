#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

  // Initialize the variables
  int mine = rank+1;
  int total = 0;

  // Print the variables before the reduction
  printf("(1) before: rank %d: mine: %d, total: %d\n", rank, mine, total);

  // Reduce
  TryMPI(MPI_Reduce(&mine, // Where in memory is the message that will be sent?
                    &total, // Where in memory should the received message be 
                            // stored?
                    1, // How many elements are in the message?
                    MPI_INT, // What is the datatype of the elements in the 
                             // message?
                    MPI_SUM, // What is the operation to be performed? 
                    0, // Who is the receiver?
                    MPI_COMM_WORLD)); // Which processes are involved in this 
                                      // communication?

  // Print the variables before the reduction
  printf("(2) after: rank %d: mine: %d, total: %d\n", rank, mine, total);

  // Finish up with MPI
  TryMPI(MPI_Finalize());

  return 0;
}

