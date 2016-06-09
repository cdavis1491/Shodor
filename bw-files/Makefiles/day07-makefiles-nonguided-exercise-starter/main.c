// SIR model - A certain number of people (PEOPLE_COUNT) move randomly in a 2D
// world of cells (with dimensions CELL_COUNT_Y by CELL_COUNT_X). More than one
// person can occupy the same cell at once. Some people start out infected;
// the rest start out susceptible. If an infected person is directly next to
// (above, to the left, to the right, or below) or on top of a susceptible
// person, the susceptible person has some percent chance of becoming infected
// (CONTAGIOUSNESS). After an infected person has been infected for a certain
// number of time steps (INFECTED_TIME_COUNT), the person becomes recovered.
// The model runs for a certain number of time steps (TIME_COUNT).
//
// Author: Aaron Weeden, Shodor, 2016

// Import libraries
#include "init.h"
#include "simulate.h"

// Define the main function
int main() {
  // Initialize the model
  Init();

  // Run the simulation
  Simulate();

  // Exit the program
  return 0;
}

