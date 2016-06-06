/**************************************************************
 * Filename: variables.c
 * Autor: Mobeen Ludin
 * Description: simple program that introduces variables and
 *      data types in C.
 *
 * How Compile: gcc -g variables.c -o variables.exe
 * How to Run: ./variables.exe
 * Output: Area of a circle with radius: 8 is: 201.06194
 *************************************************************/
#include <stdio.h> // C standard Library that defines printf function.

#define PI 3.14159265358979323846   // Defining constants

int main(){
    int radius;            // Variable declaration
    float area = 0.0;       // Variable initialization
    double pi = PI;         // C is case sensitive
    radius = 8;
    area = pi * radius * radius;

    printf("Area of a circle with radius: %d is: %.5f \n", radius, area);
    return 0;
} // END: main()
