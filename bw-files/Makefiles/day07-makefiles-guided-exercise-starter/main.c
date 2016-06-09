/* Model of a forest fire - a 2D rectangular grid of trees is initialized
   with one random tree caught on fire. At each time step, trees that are not
   on fire yet check their neighbors to the north, east, south, and west, and
   if any of them are on fire, the tree catches fire with some percent chance.
   The model runs for a certain number of time steps, which can be controlled
   by the user. At the end of the simulation, the program outputs the total
   percentage of trees burned. Tree data can also be output at each time step
   if a filename is provided by the user.
   */

/* Author: Aaron Weeden, Shodor, 2015 */

/* Naming convention:
   ALL_CAPS for constants
   CamelCase for globals and functions
   lowerCase for locals
   */
#include "utils.h"
#include "input.h"
#include "memory.h"
#include "output.h"
#include "simulation.h"

/* Declare global parameters */
int NRows = N_ROWS_DEFAULT;
int NCols = N_COLS_DEFAULT;
int BurnProb = BURN_PROB_DEFAULT;
int NMaxBurnSteps = N_MAX_BURN_STEPS_DEFAULT;
int NSteps = N_STEPS_DEFAULT;
int RandSeed = RAND_SEED_DEFAULT;
bool IsOutputtingEachStep = DEFAULT_IS_OUTPUTTING_EACH_STEP;
bool IsRandFirstTree = DEFAULT_IS_RAND_FIRST_TREE;
char *OutputFilename;

/* Declare other needed global variables */
bool AreParamsValid = true; /* Do the model parameters have valid
                                      values? */
int NTrees; /* Total number of trees in the forest */
int NRowsPlusBounds; /* Number of rows of trees plus the boundary rows */
int NColsPlusBounds; /* Number of columns of trees plus the boundary columns */
int NTreesPlusBounds; /* Total number of trees plus the boundaries */
int MiddleRow; /* The tree in the middle is here. If an even number of rows,
                  this tree is just below the middle */
int MiddleCol; /* The tree in the middle is here. If an even number of cols,
                  this tree is just to the right of the middle */
int CurStep; /* The current time step */
int NBurnedTrees; /* The total number of burned trees */
char ExeName[32]; /* The name of the program executable */
int NMaxBurnStepsDigits; /* The number of digits in the max burn steps; used for
                            outputting tree data */
int *Trees; /* 1D tree array, contains a boundary around the outside of the
               forest so the same neighbor checking algorithm can be used on
               all cells */
int *NewTrees; /* Copy of 1D tree array - used so that we don't update the
                  forest too soon as we are deciding which new trees
                  should burn -- does not contain boundary */
FILE *OutputFile; /* For outputting tree data to a file */

/* DECLARE MAIN FUNCTION */

/* @param argc The number of command line arguments
   @param argv String of command line arguments
   */
int main(int argc, char **argv) {
  /* Set the program executable name */
  strcpy(ExeName, argv[0]);

  /* Allow the user to change simulation parameters via the command line */
  GetUserOptions(argc, argv);

  if (!AreParamsValid) {
    /* Model parameters are not valid; exit early */
    PrintUsageAndExit();
  }

  if (IsOutputtingEachStep) {
    /* Open the output file */
    OutputFile = fopen(OutputFilename, "w");
  }

  /* Do some calculations before splitting up the rows */
  NTrees = NRows * NCols;
  NRowsPlusBounds = NRows + 2;
  NColsPlusBounds = NCols + 2;
  NTreesPlusBounds = NRowsPlusBounds * NColsPlusBounds;
  MiddleRow = NRows / 2;
  MiddleCol = NCols / 2;

  /* Allocate dynamic memory for the 1D tree arrays */
  AllocateMemory();

  /* Seed the random number generator */
  srandom(RandSeed);

  /* Initialize number of burned trees */
  NBurnedTrees = 0;

  /* Light a random tree on fire, set all other trees to be not burning */
  InitData();

  /* Start the simulation looping for the specified number of time steps */
  for (CurStep = 0; CurStep < NSteps; CurStep++) {
    if (IsOutputtingEachStep) {
      /* Output tree data for the current time step */
      OutputData();
    }

    /* For trees already burning, increment the number of time steps they have
       burned */
    ContinueBurning();

    /* Find trees that are not on fire yet and try to catch them on fire from
       burning neighbor trees */
    BurnNew();

    /* Copy new tree data into old tree data */
    AdvanceTime();
  }

  /* Print the total percentage of trees burned */
  printf("%.2f%% of the trees were burned\n",
      (100.0 * NBurnedTrees) / NTrees);

  if (IsOutputtingEachStep) {
    /* Close the output file */
    fclose(OutputFile);
  }

  /* Free allocated memory */
  FreeMemory();

  return 0;
}

