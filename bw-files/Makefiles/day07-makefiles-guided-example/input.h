#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Define default number of rectangles */
#define RECTS_PER_SIM_DEFAULT 10

/* Define description of input parameter for number of rectangles */
#define RECTS_PER_SIM_DESCR \
  "This many rectangles will be used (positive integer)"

/* Define character used on the command line to change the number of
   rectangles */
#define RECTS_PER_SIM_CHAR 'r'

/* Define options string used by getopt() */
extern char const GETOPT_STRING[];

extern unsigned long long int RectsPerSim;

void Input(int const argc, char * const * const argv);

