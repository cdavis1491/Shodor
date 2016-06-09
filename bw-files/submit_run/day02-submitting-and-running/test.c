#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

void TryMPI(int const err);

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  TryMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int size;
  TryMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  char name[80];
  int len;
  MPI_Get_processor_name(name, &len);
  printf("PE %2d is on core %d on node %s\n", rank, sched_getcpu(), name);
  TryMPI(MPI_Finalize());
  return 0;
}

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

