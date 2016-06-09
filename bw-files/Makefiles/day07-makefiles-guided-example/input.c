#include "input.h"

/* Define options string used by getopt() - a colon after the character means
   the parameter's value is specified by the user */
char const GETOPT_STRING[] = {
  RECTS_PER_SIM_CHAR, ':'
};

void Input(int const argc, char * const * const argv) {
  bool isError = false;
  char c;
  while ((c = getopt(argc, argv, GETOPT_STRING)) != -1) {
    switch(c) {
      /* The user has chosen to change the number of rectangles */
      case RECTS_PER_SIM_CHAR:
        /* Get integer value */
        RectsPerSim = strtoull(optarg, NULL, 10);
        /* Make sure positive and equal to floating point value */
        if (RectsPerSim < 1 || atof(optarg) != RectsPerSim) {
          fprintf(stderr, "ERROR: value for -%c must be positive integer\n",
              RECTS_PER_SIM_CHAR);
          isError = true;
        }
        break;
        /* The user has chosen an unknown option */
      default:
        isError = true;
    }
  }

  /* If there was an error in input, print a usage message and exit early */
  if (isError) {
    fprintf(stderr, "Usage: ");
    fprintf(stderr, "%s [OPTIONS]\n", argv[0]);
    fprintf(stderr, "Where OPTIONS can be any of the following:\n");
    fprintf(stderr, "-%c : \n\t%s\n\tdefault: %d\n", RECTS_PER_SIM_CHAR,
        RECTS_PER_SIM_DESCR, RECTS_PER_SIM_DEFAULT);
    exit(EXIT_FAILURE);
  }
}

