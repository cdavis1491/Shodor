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

  // Set up the array for broadcasting
  //char message[80];
  //christina makes changes 
  int message;
 
 if (rank == 0)
  {
    //strcpy(message, "Hello");
    // christina makes changes
    message = 42;
   }

  // Print the message array before the broadcast
  printf("(1) before: rank %d: %d\n", rank, message);

  // Broadcast
  TryMPI(MPI_Bcast(&message, // location in memory where the message will be 
                             // stored on all processes
                   //6, // number of elements to send/receive. 
		      // because it's "hello[null]"
                   1, // christina makes changes
		   MPI_INT, // datatype of elements to send/receive
                   0, // rank of the sending process
                   MPI_COMM_WORLD)); // communicator with group of processes 
                                     // involved in this communication

  // Print the message array after the broadcast
  printf("(2) after: rank %d: %d\n", rank, message);

  // Finish up with MPI
  TryMPI(MPI_Finalize());

  return 0;
}

