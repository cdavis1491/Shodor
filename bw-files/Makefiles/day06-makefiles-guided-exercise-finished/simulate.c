#include "simulate.h"

// Define a function to run the simulation
void Simulate() {
  for (int time = 0; time < TIME_COUNT; time++) {
    // Print the state of each cell in the environment
    PrintCells(time);

    // Move people
    MovePeople();

    // Change peoples' states
    ChangePeopleStates();
    
    // Update the model to the next time step
    Update();
  }
}

