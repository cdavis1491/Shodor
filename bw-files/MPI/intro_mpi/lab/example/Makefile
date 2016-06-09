#------- Select your Compier
COMPTYPE="Cray"
#COMPTYPE="Intel"
#COMPTYPE="GNU"
#COMPTYPE="PGI"

#------ Select known target machine
#SYSTYPE="Gordon"
#SYSTYPE="Stampede"
SYSTYPE="BW"
#SYSTYPE="Mac"
#SYSTYPE="Linux"

ifeq ($(SYSTYPE), "Mac")
CC			= gcc-5
MPICC		= mpicc
OMPFLAGS	= -fopenmp
LIBS		= -lm
endif

ifeq ($(SYSTYPE), "Gordon")
CC			= icc
MPICC		= mpicc
OMPFLAGS	= -openmp
LIBS		= -lm
endif

ifeq ($(SYSTYPE), "BW")
    ifeq ($(COMPTYPE), "Cray")
    CC			= cc
    MPICC		= cc -g
    OMPFLAGS	=
    LIBS		= -lm
    else ifeq ($(COMPTYPE), "GNU")
    CC          = gcc
    MPICC       = cc
    OMPFLAGS    = -fopenmp
    LIBS        = -lm
    endif
endif

ifeq ($(SYSTYPE), "Stampede")
CC          = icc
MPICC       = mpicc
OMPFLAGS    = -openmp
LIBS        = -lm
endif

mpi_greetings:
	$(MPICC) -o mpi_greetings.exe mpi_greetings.c
mpi_matmult:
	$(MPICC) -o mpi_matmult.exe mpi_matmult.c
all:
	make clean
	make mpi_matmult mpi_greetings

clean:
	rm -rf *.exe
