/***************************************************************
 * Filename: pointers.c
 * Author: Mobeen Ludin
 * Disreption: Introduces how to declare pointer variables and
 *  How to use them to store the address of other variables
 * How to compile:
 *      $ gcc -g pointers.c -o pointers.exe
 * How to run:
 *      $ ./pointers.exe
 ***************************************************************/
#include <stdio.h>
void main(){
    int x = 5;      // declaring an int
    int *ptr;       //declaring a integer pointer named ptr (can be anything)
    ptr = &x;       // assigning the value of ptr to the address of x
    
    // What would be the value of ptr?
    printf("ptr = %p \n", ptr);
    //What would be the output of this printf, similar to ptr or?why?
    printf("&x = %p \n", &x);
    //what would be the value of this? why different from ptr?
    printf("&ptr = %p \n", &ptr);
    //what is this? why different from &ptr?
    printf("*ptr = %d \n", *ptr); 
    // What is this pointer updating? what would be the value of x?
    *ptr = 8;
    printf("new x = %d \n", x);
    //Pointer arithmatics
    ptr = ptr +1;
    //What do you think will be the output?
    printf("ptr = %p \n", ptr); 
    printf("size of integer is: = %d bytes. \n", sizeof(int));
    // For example, if an int ptr had address 100, incrementing it by 1
    // will give the output 104, reason is its giving the address of next
    // variable, and have to skip 4 bytes to get to the new variable
}
