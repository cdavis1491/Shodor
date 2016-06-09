/**************************************************************
 * Filename: omp_functions.c
 * Description: first intro to OpenMP directives
 * How to compile:
 *  $ cc omp_functions.c -o omp_functions
 * How to set number of threads to 8:
 *  $ export OMP_NUM_THREADS=8
 * How to Run: 
 *  $ aprun -n 1 -d 8 ./omp_functions 
 *************************************************************/
#include <stdio.h>
#include <omp.h>

int main(){
    double start_t = omp_get_wtime();
    int thread_id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    int n_procs = omp_get_num_procs();
    int max_n_threads = omp_get_max_threads ( );
    double end_t = omp_get_wtime();
    
    double total_t = end_t - start_t;
    printf("System Environment Info is: \n");
    printf("\t My thread id is: %d \n", thread_id); // print thread_id
    printf("\t # of threads are: %d \n", n_threads); // 
    printf("\t max # of threads are: %d \n", max_n_threads); // 
    printf("\t $# of cores are: %d \n", n_procs);
    printf("\t Total time to get info: %lf \n", total_t);

} // END: main()
