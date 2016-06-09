#include <mpi.h>

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

