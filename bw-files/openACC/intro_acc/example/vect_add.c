/***************************************************************
 * File Name: arrays_and_pointers.c
 * Author: Mobeen Ludin
 * Date: March 29, 2016
 * 
 * How to Compile: 
 *  $ cc -h pragma=acc vect_add.c -o vect_add.exe
 * How to Run: $ aprun ./vect_add.exe
 **************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define VEC_SIZE 10000 // How big is the vector going to be.

//Function declaration
double* vector_add(double *vectA, double *vectB, double *vectSum);

int main(int argc, char *argv[]){
    struct timeval start_time, stop_time, elapsed_time;  // timers
    double *vectA;      //declaring a pointer variable named vectA
    double *vectB;      //declaring a pointer variable named vectB
    double *vectSum;    //declaring a pointer variable named vectSum
    vectA = malloc(sizeof(double) * VEC_SIZE);   //allocating memory to vectA
    vectB = malloc(sizeof(double) * VEC_SIZE);   //allocating memory to vectB
    vectSum = malloc(sizeof(double) * VEC_SIZE); //allocateding memory to vectSum
    
    int i;  // declaring loop index variable
    
    //timer starts here
    gettimeofday(&start_time,NULL);         //start time

    vector_add(vectA, vectB, vectSum);
    //stop timer
    gettimeofday(&stop_time,NULL);
    
    timersub(&stop_time, &start_time, &elapsed_time);
    printf("Total time was %f seconds.\n",
            elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    
    for(i = VEC_SIZE-10; i < VEC_SIZE; i++){
        printf("  Vector Add: A[ %.2f ] + B[ %.2f ] = Sum[ %.2f ] \n", 
                vectA[i], vectB[i], vectSum[i]);
    }
    free(vectA);    //free up the memory allocated earlier
    free(vectB);    //take back memory to use for something else
    free(vectSum);  //if dont reclaim, that memory will still be occupied.
} //End: main()

double* vector_add(double *vectorA, double *vectorB, double *vectorSum){
    int i;
#pragma acc parallel copy(vectorA[0:VEC_SIZE], vectorB[0:VEC_SIZE])
    for (i = 0; i < VEC_SIZE; i++){ 
        vectorA[i] = (i+1)*10;      //initialize array elements
        vectorB[i] = (i+1)*20;      //do same for array vectorB
    }
#pragma acc parallel copyin(vectorA[0:VEC_SIZE],\
        vectorB[0:VEC_SIZE]), copyout(vectorSum[0:VEC_SIZE])
    for ( i = 0; i < VEC_SIZE; i++){
        vectorSum[i] = vectorA[i] + vectorB[i];
    }
    return(vectorSum); 
}
