#include "output.h"

void Output() {
  /* Calculate pi and print it */
  printf("%.*f\n", DBL_DIG, 4.0 * AreaSum);
  printf("Value of pi from math.h is %.*f\n", DBL_DIG, M_PI);
}

