#include "utils.h"

// Define a function that returns a random whole number
int GetRandom(int const max) {
  return max * random() / RAND_MAX;
}

