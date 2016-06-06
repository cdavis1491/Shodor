/******************************************************************************
* Filename: omp_data_scoping.c
* Discription: Demonstrates various data scoping clauses in OpenMP.
* How to compile:
*   $ cc omp_data_scoping.c -o omp_data_scoping.c
* How set number of threads to 2:
*   $ export OMP_NUM_THREADS=2
* How to run:
*   $ aprun -n 1 -d 2 ./omp_data_scoping.exe
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int tid;
int pr=1,  //each thread will have private copy of pr
    fp=1,  //each thread will have private copy of fp=1
    sh=1,  //each thread can access this variable, shared by all.
    df=1;  //unless explicitly modified, this is always shared.

//print values before parallel region
printf("|-> Before Parallel Region:\n"
    "\tpr is: %d, \tfp is: %d \n"
    "\tsh is: %d, \tdf is: %d. \n",pr,fp,sh,df);
printf("-----------------------------------------------\n");
#pragma omp parallel shared(sh) private(pr,tid) firstprivate(fp)
  {
  tid = omp_get_thread_num();

  //print values after start of parallel region
  printf("|=> Thread %d AT START OF PARALLEL:\n"
    "\tpr is: %d, \tfp is: %d \n" 
    "\tsh is: %d, \tdf is: %d. \n",tid,pr,fp,sh,df);
  
  //update values
  pr = tid * 4; fp = pr; sh = pr; df = pr;

  //print values after update (before end of parallel region)
  printf("|=> Thread %d AFTER UPDATE:\n" 
    "\tpr is: %d, \tfp is: %d \n" 
    "\tsh is: %d, \tdf is: %d.\n",tid,pr,fp,sh,df);
  } //END: of parallel region

printf("-----------------------------------------------\n");
//print values after parallel region
printf("|-> AFTER parallel region:\n"
    "\tpr is: %d, \tfp is: %d \n"
    "\tsh is: %d, \tdf is: %d. \n",pr,fp,sh,df);
}
