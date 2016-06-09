CC 			= cc
MPICC		= cc
ifeq ($(NON_BW),1)
	CC			= gcc
	MPICC		= mpicc
	OMP_FLAGS	= -fopenmp
endif
CFLAGS		=
ifeq ($(NON_BW),1)
	CFLAGS		= -Wall
endif
MPI_FLAGS	= -D_MPI
ifeq ($(DEBUG),1)
	CFLAGS	+= -DDEBUG 
endif
ifeq ($(PROFILE),1)
	CFLAGS	+=	-pg
endif

EXPENDABLES	= serial openmp mpi-serial mpi-openmp matmul-serial.o matmul-omp.o

serial: serial-omp.c matmul-serial.o
	$(CC) $(CFLAGS) -o $@ matmul-serial.o $<

openmp: serial-omp.c matmul-omp.o
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ matmul-omp.o $<

mpi-serial: mpi.c matmul-serial.o
	$(MPICC) $(CFLAGS) $(MPI_FLAGS) -o $@ matmul-serial.o $<

mpi-openmp: mpi.c matmul-omp.o
	$(MPICC) $(CFLAGS) $(OMP_FLAGS) $(MPI_FLAGS) -o $@ matmul-omp.o $<

matmul-serial.o: matmul.c matmul.h
	$(CC) $(CFLAGS) -o $@ -c $<

matmul-omp.o: matmul.c matmul.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ -c $<

.PHONY: clean
clean:
	/bin/rm -f $(EXPENDABLES)
