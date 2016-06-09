/* Implements serial and OpenMP code for matrix-matrix multiplication
 */

#include "matmul.h"

int main(int argc,char **argv) {
    bool print = false;
    int c;
    struct matrix m1,m2,dst_m;

    unsigned int *random_seeds;

    // Initialize matrix dimensions, used in error checking later
    m1.rows = m1.cols = m2.rows = m2.cols = 0;

    // Process user parameters
    while((c = getopt(argc,argv, "x:y:a:b:p")) != -1) {
        switch(c) {
            case 'x':
                m1.cols = atoi(optarg);
                break;
            case 'y':
                m1.rows = atoi(optarg);
                break;
            case 'a':
                m2.cols = atoi(optarg);
                break;
            case 'b':
                m2.rows = atoi(optarg);
                break;
            case 'p':
                print = true;
                break;
            case '?':
                usage();
                exit(EXIT_FAILURE);
        }
    }

    if(m1.rows == 0 || m1.cols == 0 || m2.rows == 0 || m2.cols == 0) {
        fprintf(stderr,"Supply row and column counts!\n");
        usage();
        exit(EXIT_FAILURE);
    }

    // First matrix column count must match second matrix row count
    if(m1.cols != m2.rows) {
        fprintf(stderr,"Invalid matrix dimensions!\n");
        exit(EXIT_FAILURE);
    }

    // Destination matrix dimensions are equal to the first matrix's rows
    // and the second matrix's columns
    dst_m.rows = m1.rows;
    dst_m.cols = m2.cols;

    m1.matrix       = safe_malloc_int(
            m1.rows*m1.cols,
            "Allocating first matrix"
            );
    m2.matrix       = safe_malloc_int(
            m2.rows*m2.cols,
            "Allocating second matrix"
            );
    dst_m.matrix    = safe_malloc_int(
            dst_m.rows*dst_m.cols,
            "Allocating destination matrix"
            );

    // Each thread will get a separate random seed
    random_seeds = init_random_seeds();

    // Initialize source matrices
    init_matrix(&m1,random_seeds);
    init_matrix(&m2,random_seeds);

    if(print) {
        puts("Matrix 1\n");
        print_matrix(&m1);
        puts("");
        puts("Matrix 2\n");
        print_matrix(&m2);
        puts("");
    }

    // Matrix computation
    matmul(&m1,&m2,&dst_m,0,dst_m.rows);

    if(print) {
        puts("Destination matrix\n");
        print_matrix(&dst_m);
    }

    // Free heap memory
    free(m1.matrix);
    free(m2.matrix);
    free(dst_m.matrix);

    exit(EXIT_SUCCESS);
}
