/***************************************************************
 * Filename: omp_schedule_1.c
 * Author: Mobeen Ludin
 * Date: March 29, 2016
 * Discription: Simple Vector addition example to show how
 *  different openmp schedule clauses work.
 *
 * How to Compile: $ gcc omp_schedule_1.c -o omp_schedule_1.exe
 * How to Run: $ ./omp_schedule_1.exe | grep -e "Thread 0"
 **************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define CHUNK_SIZE 10
#define VEC_SIZE 100
double* vector_add(double *vectA, double *vectB, double *vectSum);

int main(){
    double *vectA;
    double *vectB;
    double *vectSum;
    vectA = malloc(sizeof(double) * VEC_SIZE);
    vectB = malloc(sizeof(double) * VEC_SIZE);
    vectSum = malloc(sizeof(double) * VEC_SIZE);
    int i;
    vector_add(vectA, vectB, vectSum);
    /*for(i = 0; i < VEC_SIZE; i++){
        printf("Vector Add: A[ %.2f ] + B[ %.2f ] = Sum[ %.2f ] \n", vectA[i], vectB[i], vectSum[i]);
    } */
    free(vectA);
    free(vectB);
    free(vectSum);
}
double* vector_add(double *vectA, double *vectB, double *vectSum){
    int i;
    int chunk = CHUNK_SIZE;
    for (i = 0; i < VEC_SIZE; i++){
        vectA[i] = (i+1)*10;
    }
    for (i=0; i < VEC_SIZE; i++){
        vectB[i] = vectA[i];
    }
    // Start of parallel region
    #pragma omp parallel shared(vectSum, vectA, vectB, chunk) private(i)
    {  
        #pragma omp for schedule(dynamic,chunk) 
        for ( i = 0; i < VEC_SIZE; i++){
            vectSum[i] = vectA[i] + vectB[i];
        printf("Thread %d Did Itteration [ %d ] : \n" 
           "Vector Add: A[ %.2f ] + B[ %.2f ] = Sum[ %.2f ] \n",
            omp_get_thread_num(), i, vectA[i], vectB[i], vectSum[i]);
        }
    } // End of Parallel Region
    return(vectSum);
}
