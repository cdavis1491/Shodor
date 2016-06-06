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
    CC			= cc -O2
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

omp_greetings:
	$(CC) $(OMPFLAGS) -o omp_greetings.exe omp_greetings.c
omp_data_scoping:
	$(CC) $(OMPFLAGS) -o omp_data_scoping.exe omp_data_scoping.c
omp_functions:
	$(CC) $(OMPFLAGS) -o omp_functions.exe omp_functions.c
omp_schedule_1:
	$(CC) $(OMPFLAGS) -o omp_schedule_1.exe omp_schedule_1.c
omp_schedule_2:
	$(CC) $(OMPFLAGS) -o omp_schedule_2.exe omp_schedule_2.c
omp_critical:
	$(CC) $(OMPFLAGS) -o omp_critical.exe omp_critical.c
omp_reduction:
	$(CC) $(OMPFLAGS) -o omp_reduction.exe omp_reduction.c

all:
	make clean
	make omp_greetings omp_data_scoping omp_functions
	make omp_schedule_1 omp_schedule_2
	make omp_critical omp_reduction

clean:
	rm -rf *.exe
