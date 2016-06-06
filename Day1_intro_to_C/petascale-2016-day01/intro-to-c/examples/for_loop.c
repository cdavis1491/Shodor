/**************************************************************
 * Filename: greetings.c
 * Autor: Mobeen Ludin
 * Disreption: simple vector addition example that shows how 
 *      for loops could be used to initialize the vector
 *      elements to some value, doing compuation, and how to
 *      print an array of fixed number of elements. For loops
 *      are basically used to repeat an operation a fixed 
 *      number of times. 
 * How Compile: gcc for_loop.c -o for_loop.exe
 * How to Run: ./for_loop.exe
 *************************************************************/
#include <stdio.h>

#define VEC_SIZE 100

int main(){
    float vectA[VEC_SIZE];
    float vectB[VEC_SIZE];
    float vectSum[VEC_SIZE];
    int i; 

    //initializing the vector elements to be as its index number.
    for(i=0; i < VEC_SIZE; i++){
        vectA[i] = vectB[i] = i * 1.0;
    }

    // adding vectA + vectB and storing results in vectC
    for(i=0; i < VEC_SIZE; i++){    
        vectSum[i] = vectA[i] + vectB[i];  
    }

    //using for loop to itterate over array and print each element.
    for(i = 0; i < VEC_SIZE; i++){ 
        printf("Vector Add: A[ %.2f ] + B[ %.2f ] = Sum[ %.2f ] \n", vectA[i], vectB[i], vectSum[i]);
    }
    return 0;
} // END: main()

