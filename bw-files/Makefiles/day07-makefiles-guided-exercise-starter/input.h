#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Define descriptions of command line options */
#define N_ROWS_DESCR \
  "The forest has this many rows of trees (positive integer)"
#define N_COLS_DESCR \
  "The forest has this many columns of trees (positive integer)"
#define BURN_PROB_DESCR \
  "Chance of catching fire if next to burning tree (positive integer [0..100])"
#define N_MAX_BURN_STEPS_DESCR \
  "A burning tree stops burning after this many time steps (positive integer bigger than 1)"
#define N_STEPS_DESCR \
  "Run for this many time steps (positive integer)"
#define RAND_SEED_DESCR \
  "Seed value for the random number generator (positive integer)"
#define OUTPUT_FILENAME_DESCR \
  "Filename to output tree data at each time step (file must not already exist)"
#define IS_RAND_FIRST_TREE_DESCR \
  "Start the first on a random first tree as opposed to the middle tree"

/* Define default values for simulation parameters - each of these parameters
   can also be changed later via user input */
#define N_ROWS_DEFAULT 21
#define N_COLS_DEFAULT N_ROWS_DEFAULT
#define BURN_PROB_DEFAULT 100
#define N_MAX_BURN_STEPS_DEFAULT 2
#define N_STEPS_DEFAULT N_ROWS_DEFAULT
#define RAND_SEED_DEFAULT 1
#define DEFAULT_IS_OUTPUTTING_EACH_STEP false
#define DEFAULT_IS_RAND_FIRST_TREE false

/* Define characters used on the command line to change the values of input
   parameters */
#define N_ROWS_CHAR 'r'
#define N_COLS_CHAR 'c'
#define BURN_PROB_CHAR 'b'
#define N_MAX_BURN_STEPS_CHAR 'm'
#define N_STEPS_CHAR 't'
#define RAND_SEED_CHAR 's'
#define OUTPUT_FILENAME_CHAR 'o'
#define IS_RAND_FIRST_TREE_CHAR 'f'

/* Define options string used by getopt() - a colon after the character means
   the parameter's value is specified by the user */
extern const char GETOPT_STRING[];

/* Declare global parameters */
extern int NRows;
extern int NCols;
extern int BurnProb;
extern int NMaxBurnSteps;
extern int NSteps;
extern int RandSeed;
extern bool IsOutputtingEachStep;
extern char *OutputFilename;
extern bool IsRandFirstTree;

/* Declare other needed global variables */
extern bool AreParamsValid; /* Do the model parameters have valid values? */
extern char ExeName[32]; /* The name of the program executable */

/* DECLARE FUNCTIONS */

/* Prints out a description of an integer command line option

   @param optChar The character used to specify the option
   @param optDescr The description of the option
   @param optDefault The default value of the option
   */
void DescribeOptionInt(const char optChar, const char *optDescr,
    const int optDefault);

/* Prints out a description of a string command line option

   @param optChar The character used to specify the option
   @param optDescr The description of the option
   */
void DescribeOptionNoDefault(const char optChar, const char *optDescr);

/* Print an error message

   @param errorMsg Buffer containing the message
   */
void PrintError(const char *errorMsg);

/* Display to the user what options are available for running the program and
   exit the program in failure 

   @param errorMsg The error message to print
   */
void PrintUsageAndExit();

/* Assert that a user's input value is an integer. If it is not, print
   a usage message and an error message and exit the program.

   @param param The user's input value
   @param optChar The character used to specify the user's input value
   */
void AssertInteger(int param, const char optChar);

/* Assert that a user's input value is a positive integer. If it is not, print
   a usage message and an error message and exit the program.

   @param param The user's input value
   @param optChar The character used the specify the user's input value
   */
void AssertPositiveInteger(int param, const char optChar);

/* Assert that a user's input value is bigger than a value. If it is
   not, print a usage message and an error message and exit the program.

   @param param The user's input value
   @param low The value the parameter needs to be bigger than
   @param optChar The character used the specify the user's input value
   */
void AssertBigger(int param, const int val, const char optChar);

/* Assert that a user's input value is between two values, inclusive. If it is
   not, print a usage message and an error message and exit the program.

   @param param The user's input value
   @param low The lowest value the parameter can be
   @param high The highest value the parameter can be
   @param optChar The character used the specify the user's input value
   */
void AssertBetweenInclusive(int param, const int low, const int high,
    const char optChar);

/* Exit if a file already exists */
void AssertFileDNE(const char *filename);

/* Allow the user to change simulation parameters via the command line

   @param argc The number of command line arguments to parse
   @param argv The array of command line arguments to parse
   */
void GetUserOptions(const int argc, char **argv);

