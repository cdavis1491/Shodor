/**************************************************************
 * Filename: omp_schedule.c
 * Description: vector addition example to show schedule clauses.
 * How to compile:
 *  $ cc omp_schedule.c -o omp_schedule
 * How to set number of threads to 8:
 *  $ export OMP_NUM_THREADS=8
 * How to Run: 
 *  $ aprun -n 1 -d 8 ./omp_schedule 
 *************************************************************/
#include <stdio.h>
#include <stdlib.h>

#define VEC_SIZE 80000000 // How big is the vector going to be.
#define CHUNK_SIZE 10000
//Function declaration
double* vector_add(double *vectA, double *vectB, double *vectSum);

//main takes two arguments: argc to store the number of args passed to the
//program. argv[] is pointer to array of characters. It stores the actual
//arrguments passed to the program in order.
int main(int argc, char *argv[]){

    double *vectA;      //declaring a pointer variable named vectA
    double *vectB;      //declaring a pointer variable named vectB
    double *vectSum;    //declaring a pointer variable named vectSum
    vectA = malloc(sizeof(double) * VEC_SIZE);   //allocating memory to vectA
    vectB = malloc(sizeof(double) * VEC_SIZE);   //allocating memory to vectB
    vectSum = malloc(sizeof(double) * VEC_SIZE); //allocateding memory to vectSum
    int i;  // declaring loop index variable

    //from inside main calling vector_add() function and passing it three vectors
    //as arguments. 
    vector_add(vectA, vectB, vectSum);
    
    // printing values stored in vectSum[]; 
    for(i = VEC_SIZE-10; i < VEC_SIZE; i++){
        printf("Vector Add: A[ %.2f ] + B[ %.2f ] = Sum[ %.2f ] \n", vectA[i], vectB[i], vectSum[i]);
    }

    free(vectA);    //free up the memory allocated earlier
    free(vectB);    //take back memory to use for something else
    free(vectSum);  //if dont reclaim, that memory will still be occupied.
} //End: main()

// vector_add function initialization. Its arguments are pass by reference
//Remember these are still new variables and will occupy memory. Except its
//content will be address to vectA, vectB, and vectSum. 
double* vector_add(double *vectorA, double *vectorB, double *vectorSum){
    
    int i; //Remember all variables inside functions are local, so do again.

#pragma omp parallel private(i) shared(vectorA, vectorB,vectorSum)
    {
       #pragma omp for schedule(static, CHUNK_SIZE)
        for (i = 0; i < VEC_SIZE; i++){ 
            vectorA[i] = (i+1)*10;      //initialize array elements
            vectorB[i] = (i+1)*20;      //do same for array vectorB
        }   
        #pragma omp for schedule(dynamic,CHUNK_SIZE)
            for ( i = 0; i < VEC_SIZE; i++){
                vectorSum[i] = vectorA[i] + vectorB[i];            
            }
    }

    return(vectorSum); //function returns a double, and will return sum.
}
