/* This file contains the "meat" of the matrix-matrix multiplication. It
 * can be run serially, or with OpenMP support. The matmul function is designed
 * to be called from MPI, although it has no MPI support on its own.
 */

#include "matmul.h"

void usage() {
    fprintf(stderr,"serial -x <m1-cols> -y <m1-rows> -a <m2-cols> -b <m2-rows>  [ -p ] \n");
    fprintf(stderr,"Supply -p if you want matrices printed out\n");
}

// Take in an integer c and a message to print on failure
// Allocate an array of c ints and return it
int *safe_malloc_int(const unsigned int c,const char *msg) {
    int *m;
    if((m = malloc(c*sizeof(int))) == NULL) {
        fprintf(stderr,"%s failed at %s, line %d: %s\n",
                msg,__FILE__,__LINE__,strerror(errno));
        exit(EXIT_FAILURE);
    }

    return m;
}

// Take in an unsigned integer c and a message to prunsigned int on failure
// Allocate an array of c unsigned ints and return it
unsigned int *safe_malloc_unsigned_int(const unsigned int c,const char *msg) {
    unsigned int *m;
    if((m = malloc(c*sizeof(unsigned int))) == NULL) {
        fprintf(stderr,"%s failed at %s, line %d: %s\n",
                msg,__FILE__,__LINE__,strerror(errno));
        exit(EXIT_FAILURE);
    }

    return m;
}

// Return an array of random number seeds. Epoch time is used as a base, incremented by thread ID
unsigned int *init_random_seeds() {
    unsigned int i,*seeds,num_threads;
    time_t t = time(NULL);
#ifdef _OPENMP
#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
#else
    num_threads = 1;
#endif

    seeds = safe_malloc_unsigned_int(num_threads,"Allocating random seed array");

#ifdef DEBUG
    fprintf(stderr,"Setting seed for %u threads\n",num_threads);
#endif
    for(i=0;i<num_threads;i++) {
        // Guarantee different starting seeds for random numbers in each thread
        seeds[i] = t+i;
#ifdef DEBUG
        fprintf(stderr,"seeds[%d] = %u\n",i,seeds[i]);
#endif
    }

    return seeds;
}

// Takes in a matrix, and a pointer to an array of random seeds.
// Sets a random number in each cell
void init_matrix(struct matrix *m,unsigned int *s) {
    int i,t;

#ifdef _OPENMP
#pragma omp parallel 
    {
        // Initialize t to thread number if using OpenMP
    t = omp_get_thread_num();
#pragma omp for schedule(static,1)
#else
    // When not using OpenMP, only one "thread" will be running
    t = 0;
#endif
    for(i=0;i<m->rows*m->cols;i++) {
        // Limit matrix values < 10
        m->matrix[i] = rand_r(&s[t]) % 10;
    }
#ifdef _OPENMP
    }
#endif
}

// Takes in a reference to a matrix and a row/column coordinate
// Returns the one-dimensional index in the matrix for the coordinate
unsigned int coord(const struct matrix *m,const unsigned int row,const unsigned int col) {
    return row*m->cols+col;
}

// Takes in a matrix and prints it to STDERR
void print_matrix(const struct matrix *m) {
    int row,col;

    for(row=0;row<m->rows;row++) {
        for(col=0;col<m->cols;col++) {
            fprintf(stderr,"%d ",m->matrix[coord(m,row,col)]);
        }
        fprintf(stderr,"\n");
    }
}

#ifndef _OPENMP
struct timeval safe_gettimeofday() {
    struct timeval now_t;

    if((gettimeofday(&now_t,NULL)) == -1) {
        fprintf(stderr,"gettimeofday failed: %s\n",strerror(errno));
        exit(EXIT_FAILURE);
    }

    return now_t;
}
#endif

void matmul(
        struct matrix *m1,
        struct matrix *m2,
        struct matrix *dst_m,
        const unsigned int start_dst_row, // Start generating at this row of the product matrix
        const unsigned int end_dst_row // End generating at this row of the product matrix
        ) {
    unsigned int dst_row,dst_col,i,dst_coord;

    // Parallelize outer two loops. Make sure destination coordinate
    // and loop for array calculation have thread-local variables
#ifdef _OPENMP
    double start_t = omp_get_wtime();
#pragma omp parallel for private(dst_coord,i) collapse(2) schedule(static,1)
#else
    struct timeval start_t = safe_gettimeofday();
#endif
    // Process each cell in the destination matrix, and calculate the result
    // based on the two source matrices
    for(dst_row=start_dst_row;dst_row<end_dst_row;dst_row++) {
        for(dst_col=0;dst_col<dst_m->cols;dst_col++) {
#ifdef DEBUG
            fprintf(stderr,"Calculating (%d,%d)\n",dst_row,dst_col);
#endif
            /* Assign destination coordinates based on the start row offset
             * For serial/OpenMP, this has no effect since the full destination
             * matrix will be used
             * For MPI, a smaller, local destination matrix will be used and the
             * indexes must start at 0 regardless of how far into the 
             * destination matrix we are
            */
            dst_coord = coord(dst_m,(dst_row-start_dst_row),dst_col);
            // Make sure destination matrix is initialized
            dst_m->matrix[dst_coord] = 0;
            /* 
             * The number of rows in m1 is guaranteed to be the number
             * of columns in m2, so when we are done processing m1's rows
             * we are also done processing m2's columns
             * This part is not as amenable to parallelization since the
             * multiple threads would have to update the cell in the
             * product matrix at the same time
            */
            for(i=0;i<m1->rows;i++) {
                /* 
                 * The destination matrix cell will accumulate pair-wise
                 * multiplication results from every cell in the current
                 * row in the first matrix, and every cell in the current
                 * column in the second matrix
                 */
                dst_m->matrix[dst_coord] +=
                    m1->matrix[coord(m1,dst_row,i)] 
                    * m2->matrix[coord(m2,i,dst_col)];
            }
#ifdef DEBUG
            fprintf(stderr,"m1(%d,%d) * m2(%d,%d) = dst_m(%d,%d) (%d)\n",
                    dst_row,i,i,dst_col,dst_row,dst_col,
                    dst_m->matrix[dst_coord]
                   );
#endif
        }
    }
#ifdef _OPENMP
    fprintf(stderr,"OpenMP took %f seconds\n",omp_get_wtime()-start_t);
#else
    struct timeval end_t,run_t;
    end_t = safe_gettimeofday();
    timersub(&end_t,&start_t,&run_t);
    fprintf(stderr,"Serial took %ld.%ld seconds\n",
            (long int)run_t.tv_sec,(long int)run_t.tv_usec);
#endif
}
