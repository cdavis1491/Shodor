/**********************************************************************
 * Filename: functions.c
 * Author: Mobeen Ludin
 * Discription: Shows how functions are created in C, and how they are
 *      called from other functions, and how arguments are passed
 * How to Compile: $ gcc functions.c -o functions.exe
 * How to Run: $ ./functions.exe
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>

// Functions declaration here: 
// Similar to a variable a function must be declared first before used.
int addition(float, float);
float subtraction(float, float);
float division(float, float);
float multiplication(float, float);

//Global variables. Should be only variable used by many functions.
float result = 0;

int main() {
    float number1 = 0;
    float number2 = 0;
    int choice;

    //put() is similar to printf, but can only print strings. 
    puts("----------------------The Calculator---------------------");
    printf("Please chose one of the following\n"
         "\t 1. Addition       \t \t 2. Subtraction\n"
         "\t 3. Multiplication \t \t 4. Divission\n");
    puts("---------------------------------------------------------");
    //scanf() used to prompt the user for input.
    scanf("%d", &choice);

    puts("Please enter number 1:");
    scanf("%f", &number1);
    puts("Please enter number 2:");
    scanf("%f", &number2);

    // Do the math as requested
    switch(choice)
    {
        case 1:
            result = addition(100, 200);
            printf("addition of passing two numbers is: %f \n", result);
            printf("Computing %.2f + %.2f = %.d \n", number1, number2, addition(number1, number2));
            break;  // if the condition was true, dont bother checking rest. 
        case 2:
            subtraction(number1, number2);
            break;
        case 3:
            result = multiplication(number1, number2);
            printf("Computing: %.2f * %.2f = %.2f \n", number1, number2, result );
            break;
        case 4:
            division(number1, number2);
            break;
        default:
            printf("Error: Please check numbers are valid \n");
            exit(1); // quit the program, cos the necessary inputs are wrong.
    } // END OF SWITCH
    printf("Thank you, Come Again \n");
} // END OF MAIN

// Addition Functioon
int addition(float num1, float num2){
    int sum = 0;
    result = num1 + num2;
    return sum;
} 

// Subtraction Function
float subtraction(float num1, float num2) {
    //instead of doing print in main, let the function do that too. 
    printf("Computing: %.2f - %.2f = %.2f \n", num1, num2, result = num1 - num2);
    return result;
}

// Multiplication Function
float multiplication(float num1, float num2){
    result = num1 * num2;
    //printf("Computing: %.2f * %.2f = %.2f \n", num1, num2, result); 
    return result;
}

// Division Function
float division(float num1, float num2){
    // Lets check to make sure we are not dividing by zero
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

