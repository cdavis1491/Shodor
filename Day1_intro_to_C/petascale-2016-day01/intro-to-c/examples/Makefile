#------- Select your Compier
#COMPTYPE="Cray"
#COMPTYPE="Intel"
COMPTYPE="GNU"
#COMPTYPE="PGI"

#------ Select known target machine
#SYSTYPE="Gordon"
#SYSTYPE="Stampede"
#SYSTYPE="BW"
#SYSTYPE="Mac"
SYSTYPE="Linux"

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

ifeq ($(SYSTYPE), "Linux")
    ifeq ($(COMPTYPE), "intel")
    CC			= icc
    MPICC		= mpicc
    OMPFLAGS	= -openmp
    LIBS		= -lm
    else ifeq ($(COMPTYPE), "GNU")
    CC          = gcc
    MPICC       = mpicc
    OMPFLAGS    = -fopenmp
    LIBS        = -lm
    endif
endif

greetings:
	$(CC) -o greetings.exe greetings.c
variables:
	$(CC) -o variables.exe variables.c
conditionals:
	$(CC) -o conditionals.exe conditionals.c
for_loop:
	$(CC) -o for_loop.exe for_loop.c
functions:
	$(CC) -o functions.exe functions.c
arrays:
	$(CC) -o arrays.exe arrays.c
pointers:
	$(CC) -o pointers.exe pointers.c
pointers_and_arrays:
	$(CC) -o pointers_and_arrays.exe pointers_and_arrays.c
all:
	make clean
	make greetings variables conditionals for_loop functions
	make arrays pointers pointers_and_arrays
clean:
	rm -rf *.exe
