/**********************************************************************
 * Filename: calc.c (requires headerfile calc.h)
 * Program: Simple calculator in C could be extended.
 * Author: Mobeen Ludin
 * Date: March 23, 2016
 * 
 * How to Compile: gcc -Wall -ggdb calc.c -o calc.exe
 *
 * How to Run: ./calc.exe
 *
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "calc.h"

int main() {
    float num1 = 0;
    float num2 = 0;
    int choice;
    bool checker = false;
    
    puts("|----------------------The Calculator----------------------");
    printf("| Please chose one of the following: [ENTER A NUMBER]\n"
    "| 1. Addition             \t  2. Subtraction\n"
    "| 3. Multiplication       \t  4. Divission\n"
    "| 5. Fibonacci            \t  6. Computing Pi\n"
    "| 7. Temp F to C          \t  8. Compound Interest\n"
    "| 9. Root of Q.E.         \t  10. Income Tax\n"
    "| 11. Monte C. Integration\t  12. Finding N-Series \n");
    puts("|----------------------------------------------------------");
 // Make sure the right option for computation is entered
 do {
    scanf("%d", &choice);
    if ((choice < 1) || (choice > 6)){
        printf("Please choose the correct type of computation\n");
        exit(1);
    }
    else {
        checker = true;
    }
    } while(!checker); // END OF DO WHILE


// Do the math requested
 switch(choice)
 {
     case 1:
        printf("Computing %.2f + %.2f = %.2f \n", num1, num2, addition(num1, num2));
        break;
     case 2:
        subtraction(num1, num2);
        break;
     case 3:
        multiplication(num1, num2);
        break;
     case 4:
        division(num1, num2);
        break;
     case 5:
        for (i=0; i <= nterm; i++){
            printf("fibonacci number is: %.5f\n", fibonacci(i));
        }
        break;
     case 6:
         tempConvert();
         break;
     default:
        printf("Error: Please check numbers are valid \n");
        exit(1);

 } // END OF SWITCH
 printf("Thank you, Come Again \n");

} // END OF MAIN

// Fibonacci Number Series
float fibonacci(int nterm){ 
    if(nterm == 0)
        return (0);
    if(nterm == 1)
        return (1);
    else
        return (fibonacci(nterm-1) + fibonacci(nterm-2));
}

// Addition 
float addition(float num1, float num2){
    puts("Please enter number 1:");
    scanf("%f", &num1);
    puts("Please enter number 2:");
    scanf("%f", &num2);
    float result = 0;
    result = num1 + num2;
    return result;
} 

// Subtraction
float subtraction(float num1, float num2) {
    puts("Please enter number 1:");
    scanf("%f", &num1);
    puts("Please enter number 2:");
    scanf("%f", &num2);
    float result = 0;
    printf("Computing: %.2f - %.2f = %.2f \n", num1, num2, result = num1 - num2);
    return result;
}

// Multiplication
float multiplication(float num1, float num2){
    puts("Please enter number 1:");
    scanf("%f", &num1);
    puts("Please enter number 2:");
    scanf("%f", &num2);
    float result = 0;
    result = num1 * num2;
    printf("Computing: %.2f * %.2f = %.2f \n", num1, num2, result); 
    return result;
}

// Division
float division(float num1, float num2){
    puts("Please enter number 1:");
    scanf("%f", &num1);
    puts("Please enter number 2:");
    scanf("%f", &num2);
    float result = 0;
    if (num2 != 0){
        result = num1 / num2;
        printf("Computing: %.2f / %.2f = %.2f \n", num1, num2, result);
    }
    else{
        puts("Dividint by Zero");
    }
    return result;
}

// Computing Sin of a function
float _sin(){
    return 0;
}

float tempConvert(){
    float fahr, celsius;
    printf("Enter the value for celsius:");
    scanf("%f", &celsius);
    fahr =(9.0/5.0) * celsius + 32;
    printf("%.3fc is equal to %.3fF\n", celsius, fahr);
    return 0;
}

