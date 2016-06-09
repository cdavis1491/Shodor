# Makefile
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
    ACC         = cc
    ACCFLAGS    = -h pragma=acc,msgs
    MPICC		= cc
    OMPFLAGS	=
    LIBS		= -lm
    else ifeq ($(COMPTYPE), "PGI")
    CC          = cc
    MPICC       = 
    ACC         = cc -acc -Minfo
    ACCFLAGS    = -h pragma=acc,msgs
    OMPFLAGS    = -fopenmp
    LIBS        = -lm
    endif
endif

laplace_serial:
	$(CC) -o laplace_serial.exe laplace_serial.c   
laplace_bad_acc:
	$(ACC) $(ACCFLAGS) -o laplace_bad_acc.exe laplace_bad_acc.c
laplace_acc:
	$(ACC) $(ACCFLAGS) -o laplace_acc.exe laplace_acc.c
vect_add:
	$(ACC) $(ACCFLAGS) -o vect_add.exe vect_add.c
all:
	make clean
	make laplace_serial laplace_bad_acc vect_add
clean:
	rm -rf *.exe *.ptx *.cub
