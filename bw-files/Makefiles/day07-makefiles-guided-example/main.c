/* Approximate pi using a Left Riemann Sum under a quarter unit circle.
   Author: Aaron Weeden, Shodor, 2015
 */
#include "input.h"
#include "simulate.h"
#include "output.h"

unsigned long long int RectsPerSim = RECTS_PER_SIM_DEFAULT;
double AreaSum = 0.0;

int main(int argc, char **argv) {
  Input(argc, argv);

  Simulate();

  Output();

  return 0;
}

