#include <stdio.h>
#include "utils.h"

/* Declare global parameters */
extern int NRows;
extern int NCols;
extern int NMaxBurnStepsDigits;

/* Declare other needed global variables */
extern int NColsPlusBounds;
extern int CurStep;
extern int *Trees;
extern FILE *OutputFile;

/* Output tree data for the current time step */
void OutputData();

