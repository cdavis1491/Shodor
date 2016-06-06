/**************************************************************
 * Filename: omp_greetings.c
 * Description: first intro to OpenMP directives
 * How to compile:
 *  $ cc omp_greetings.c -o omp_greetings.exe
 * How to set number of threads to 8:
 *  $ export OMP_NUM_THREADS=8
 * How to Run: 
 *  $ aprun -n 1 -d 8 ./omp_greetings.exe 
 *************************************************************/
#include <stdio.h>      // Used for printf()
#include <omp.h>        // used for OpenMP routines

int main(){
    int tid;
    tid = omp_get_thread_num();
    printf("|-> Greetings from Master Thread: %d \n", tid);
    
#pragma omp parallel
    {
        tid = omp_get_thread_num();
        printf("\t|=> Greetings from worker thread: %d.\n", tid);
    
    } //END: omp parallel 

    printf("|-> Back to just master thread (): Goodbye. \n");
} // END: main()

