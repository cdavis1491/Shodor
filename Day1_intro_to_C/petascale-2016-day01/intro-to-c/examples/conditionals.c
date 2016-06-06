/**************************************************************
 * Filename: conditionals.c
 * Autor: Mobeen Ludin
 * Description: simple program that shows how to use if and else
 *      statements as conditionals.
 *
 * How Compile: gcc -g conditionals.c -o conditionals.exe
 * How to Run: ./conditionals.exe
 * Output: 
 *  What would like to solve for in circlue:
 *      1. area     2. diameter     3. circumference
 *   2
 *   What is the radius of the circle?
 *   12
 *     Diameter of a circle with radius 12 is: 24.00000
 *************************************************************/
#include <stdio.h> // C standard Library that defines printf function.

#define PI 3.14159265358979323846   // Defining constants

int main(){
    int radius;             //
    float area = 0.0;       // A = π * r^2
    float diameter;         // d = 2 * r
    float circumference;    // C = 2 * π * r
    double pi = PI;         // C is case sensitive
    
    int solve_for;
    printf("What would like to solve for in circlue:\n"
            "\t 1. area \t 2. diameter \t 3. circumference \n");
    scanf("%d", &solve_for);
    
    printf("What is the radius of the circle?\n");
    // scanf() C standard input, prompts for user input
    scanf("%d", &radius);
    
    //lets evaluate users input, what to solve for?
    if (solve_for == 1){
        area = pi * radius * radius;
        printf("\tArea of a circle with radius: %d is: %.5f \n", radius, area);
    }
    else if (solve_for == 2){
        diameter = 2 * radius;
        printf("\t Diameter of a circle with radius %d is: %.5f \n", radius, diameter);
    }
    else if (solve_for == 3){
        circumference = 2 * pi * radius;
        printf("Circumference with radius: %d is: %.5f \n", radius, circumference);
    }
    else {
        printf("I have no idea what are you trying to do\n");
    }
    return 0;

} // END: main()
