# Parallelization:  Sieve of Eratosthenes
# By Aaron Weeden, Shodor Education Foundation, Inc.
# January 2012
#
# Makefile
#
# Usage:  On the command line, enter 'make' followed by one of the options below
# e.g. $ make serial

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

ifeq ($(SYSTYPE), "Gordon")
CC			= icc
MPICC		= mpicc
OMPFLAGS	= -openmp
LIBS		= -lm
endif

ifeq ($(SYSTYPE), "BW")
    ifeq ($(COMPTYPE), "Cray")
    CC			= cc
    MPICC		= cc
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
	$(MPICC) -o mpi_greetings.exe mpi_greetings.c -lm   
mpi_sieve:
	$(MPICC) -o mpi_sieve.exe mpi_sieve.c -lm
all:
	make clean
	make mpi_greetings mpi_sieve
clean:
	rm -rf *.exe
