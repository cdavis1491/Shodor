#include "simulate.h"

void Simulate() {
  /* Calculate the width of each rectangle */
  double const width = (double)1 / RectsPerSim;

  /* Sum areas of all rectangles */
  for (int i = 0; i < RectsPerSim; i++) {
    /* Calculate the x-coordinate of the rectangle's left side */
    double const x = i * width;

    /* Use the circle equation to calculate the rectangle's height squared */
    double const heightSq = 1.0 - x * x;

    /* If the height squared is so close to zero that the sqrt() function would
       return -inf, do not call the sqrt() function, just set the height to zero
       */
    double const height = (heightSq < DBL_EPSILON) ? 0.0 : sqrt(heightSq);

    /* Calculate the area of the rectangle and add it to the total */
    AreaSum += width * height;
  }
}

