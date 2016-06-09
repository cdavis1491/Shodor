#ifndef MATMUL_H
#define MATMUL_H

#include <stdlib.h> // malloc(3), calloc(3)
#include <stdio.h> // printf(3)
#include <unistd.h> // exit(2)
#include <string.h> // strerror(3)
#include <errno.h> // errno mappings
#include <getopt.h> // getopt(3)
#include <time.h> // time(2)
#include <stdbool.h> // bool

#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h> // gettimeofday(2)
#endif

// Structure defining a matrix and its attributes
struct matrix {
    int *matrix;
    unsigned int rows;
    unsigned int cols;
};

void usage();

// Take in a count of objects and a failure message, returns allocated array
int *safe_malloc_int(const unsigned int,const char *);
unsigned int *safe_malloc_unsigned_int(const unsigned int,const char *);
// Returns allocated zero'd random seed array
int *init_rand_seeds();

// Returns an unsigned int array of random seeds, one per thread
unsigned int *init_random_seeds();

// Take in a reference to an allocated array its size, and a random seed. Initializes with random integers
void init_matrix(struct matrix *,unsigned int *);

// Takes in a matrix and row/column coordinate. Returns the one-dimensional
// index into the given matrix.
unsigned int coord(const struct matrix *,const unsigned int,const unsigned int);

// Takes in a matrix. Prints it to STDERR
void print_matrix(const struct matrix *);

#ifndef _OPENMP
// Returns a timeval struct for the current time, only needed for serial
struct timeval safe_gettimeofday();
#endif

// Takes in two sources matrices and a destination matrix
// Populates destination matrix by multiplying the first two matrices
void matmul(struct matrix *,struct matrix *,struct matrix *,const unsigned int,const unsigned int);

#endif
