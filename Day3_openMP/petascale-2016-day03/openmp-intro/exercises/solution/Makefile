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
    MPICC		= mpicc
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

omp_sieve:
	$(CC) $(OMPFLAGS) -o omp_sieve.exe omp_sieve.c -lm
all:
	make clean
	make omp_sieve
clean:
	rm -rf *.exe
