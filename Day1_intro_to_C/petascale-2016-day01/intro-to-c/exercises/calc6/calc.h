/**************************************************************
 * File: calc.h
 * Author: Mobeen Ludin
 * Date: March 23, 2016
 * Discription: Header file for the simple calculator program.
 *  This file only has the declaration of data types and 
 *  functions.
 * 
 *************************************************************/
// Define boolian data type
typedef enum
{
    false = 0,
    true = 1
}bool;

int nterm = 30;
int i;

// Functions declaration here:
float tempConvert();
float fibonacci(int);
float addition(float, float);
float subtraction(float, float);
float division(float, float);
float multiplication(float, float);
