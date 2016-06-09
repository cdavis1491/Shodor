#include "input.h"

/* Define options string used by getopt() - a colon after the character means
   the parameter's value is specified by the user */
const char GETOPT_STRING[] = {
  N_ROWS_CHAR, ':',
  N_COLS_CHAR, ':',
  BURN_PROB_CHAR, ':',
  N_MAX_BURN_STEPS_CHAR, ':',
  N_STEPS_CHAR, ':',
  RAND_SEED_CHAR, ':',
  OUTPUT_FILENAME_CHAR, ':',
  IS_RAND_FIRST_TREE_CHAR
};

/* DECLARE FUNCTIONS */

/* Prints out a description of an integer command line option

   @param optChar The character used to specify the option
   @param optDescr The description of the option
   @param optDefault The default value of the option
   */
void DescribeOptionInt(const char optChar, const char *optDescr,
    const int optDefault) {
  fprintf(stderr, "-%c : \n\t%s\n\tdefault: %d\n", optChar, optDescr,
      optDefault);
}

/* Prints out a description of a string command line option

   @param optChar The character used to specify the option
   @param optDescr The description of the option
   */
void DescribeOptionNoDefault(const char optChar, const char *optDescr) {
  fprintf(stderr, "-%c : \n\t%s\n", optChar, optDescr);
}

/* Print an error message

   @param errorMsg Buffer containing the message
   */
void PrintError(const char *errorMsg) {
  fprintf(stderr, "%s", errorMsg);
  AreParamsValid = false;
}

/* Display to the user what options are available for running the program and
   exit the program in failure 

   @param errorMsg The error message to print
   */
void PrintUsageAndExit() {
  fprintf(stderr, "Usage: ");
  fprintf(stderr, "%s [OPTIONS]\n", ExeName);
  fprintf(stderr, "Where OPTIONS can be any of the following:\n");
  DescribeOptionInt(N_ROWS_CHAR, N_ROWS_DESCR, N_ROWS_DEFAULT);
  DescribeOptionInt(N_COLS_CHAR, N_COLS_DESCR, N_COLS_DEFAULT);
  DescribeOptionInt(BURN_PROB_CHAR, BURN_PROB_DESCR, BURN_PROB_DEFAULT);
  DescribeOptionInt(N_MAX_BURN_STEPS_CHAR, N_MAX_BURN_STEPS_DESCR,
      N_MAX_BURN_STEPS_DEFAULT);
  DescribeOptionInt(N_STEPS_CHAR, N_STEPS_DESCR, N_STEPS_DEFAULT);
  DescribeOptionInt(RAND_SEED_CHAR, RAND_SEED_DESCR, RAND_SEED_DEFAULT);
  DescribeOptionNoDefault(OUTPUT_FILENAME_CHAR, OUTPUT_FILENAME_DESCR);
  DescribeOptionNoDefault(IS_RAND_FIRST_TREE_CHAR,
      IS_RAND_FIRST_TREE_DESCR);
  exit(EXIT_FAILURE);
}

/* Assert that a user's input value is an integer. If it is not, print
   a usage message and an error message and exit the program.

   @param param The user's input value
   @param optChar The character used to specify the user's input value
   */
void AssertInteger(int param, const char optChar) {
  char errorStr[64];

  /* Get the user's input value, assume floating point */
  const float floatParam = atof(optarg);

  /* Make sure positive and integer */
  if (floatParam != param) {
    sprintf(errorStr, "ERROR: value for -%c must be an integer\n",
        optChar);
    PrintError(errorStr);
  }
}

/* Assert that a user's input value is a positive integer. If it is not, print
   a usage message and an error message and exit the program.

   @param param The user's input value
   @param optChar The character used the specify the user's input value
   */
void AssertPositiveInteger(int param, const char optChar) {
  char errorStr[64];

  /* Get the user's input value, assume floating point */
  const float floatParam = atof(optarg);

  /* Make sure positive and integer */
  if (param < 1 || floatParam != param) {
    sprintf(errorStr, "ERROR: value for -%c must be positive integer\n",
        optChar);
    PrintError(errorStr);
  }
}

/* Assert that a user's input value is bigger than a value. If it is
   not, print a usage message and an error message and exit the program.

   @param param The user's input value
   @param low The value the parameter needs to be bigger than
   @param optChar The character used the specify the user's input value
   */
void AssertBigger(int param, const int val, const char optChar) {
  char errorStr[64];

  if (param <= val) {
    sprintf(errorStr,
        "ERROR: value for -%c must be bigger than %d\n", optChar, val);
    PrintError(errorStr);
  }
}

/* Assert that a user's input value is between two values, inclusive. If it is
   not, print a usage message and an error message and exit the program.

   @param param The user's input value
   @param low The lowest value the parameter can be
   @param high The highest value the parameter can be
   @param optChar The character used the specify the user's input value
   */
void AssertBetweenInclusive(int param, const int low, const int high,
    const char optChar) {
  char errorStr[64];

  if (param < low || param > high) {
    sprintf(errorStr,
        "ERROR: value for -%c must be between %d and %d, inclusive\n",
        optChar, low, high);
    PrintError(errorStr);
  }
}

/* Exit if a file already exists */
void AssertFileDNE(const char *filename) {
  char errorStr[64];

  if (access(filename, F_OK) != -1) {
    sprintf(errorStr,
        "ERROR: File '%s' already exists\n", filename);
    PrintError(errorStr);
  }
}


/* Allow the user to change simulation parameters via the command line

   @param argc The number of command line arguments to parse
   @param argv The array of command line arguments to parse
   */
void GetUserOptions(const int argc, char **argv) {
  char c; /* Loop control variable */

  /* Loop over argv, setting parameter values given */
  while ((c = getopt(argc, argv, GETOPT_STRING)) != -1) {
    switch(c) {
      case N_ROWS_CHAR:
        NRows = atoi(optarg);
        AssertPositiveInteger(NRows, N_ROWS_CHAR);
        break;
      case N_COLS_CHAR:
        NCols = atoi(optarg);
        AssertPositiveInteger(NCols, N_COLS_CHAR);
        break;
      case BURN_PROB_CHAR:
        BurnProb = atoi(optarg);
        AssertInteger(BurnProb, BURN_PROB_CHAR);
        AssertBetweenInclusive(BurnProb, 0, 100, BURN_PROB_CHAR);
        break;
      case N_MAX_BURN_STEPS_CHAR:
        NMaxBurnSteps = atoi(optarg);
        AssertPositiveInteger(NMaxBurnSteps, N_MAX_BURN_STEPS_CHAR);
        AssertBigger(NMaxBurnSteps, 1, N_MAX_BURN_STEPS_CHAR);
        break;
      case N_STEPS_CHAR:
        NSteps = atoi(optarg);
        AssertPositiveInteger(NSteps, N_STEPS_CHAR);
        break;
      case RAND_SEED_CHAR:
        RandSeed = atoi(optarg);
        AssertPositiveInteger(RandSeed, RAND_SEED_CHAR);
        break;
      case OUTPUT_FILENAME_CHAR:
        IsOutputtingEachStep = true;
        OutputFilename = optarg;
        break;
      case IS_RAND_FIRST_TREE_CHAR:
        IsRandFirstTree = true;
        break;
      case '?':
      default:
        PrintError("ERROR: illegal option\n");
        PrintUsageAndExit();
    }
  }

  if (IsOutputtingEachStep) {
    /* Make sure the output file does not exist (DNE) */
    AssertFileDNE(OutputFilename);
  }
}

