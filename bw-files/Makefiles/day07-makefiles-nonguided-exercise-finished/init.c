#include "init.h"

// Define a function to initialize the model
void Init() {
  // Seed the random number generator
  srandom(RANDOM_SEED);

  // Initialize the array of cell data
  InitCells();

  // Initialize the arrays of people data and the cells where people are
  InitPeople();
}

// Define a function to initialize the array of cell data
void InitCells() {
  for (int y = 0; y < CELL_COUNT_Y; y++) {                                      
    for (int x = 0; x < CELL_COUNT_X; x++) {                                    
      CellStates[y][x] = EMPTY;                                                 
    }                                                                           
  }                                                                             
}

// Define a function to initialize the arrays of people data and the cells where
// people are
void InitPeople() {
  for (int i = 0; i < PEOPLE_COUNT; i++) {                                      
    PeopleStates[i] = (i < INITIAL_INFECTED_COUNT) ? INFECTED : SUSCEPTIBLE;    
    PeopleStatesNew[i] = PeopleStates[i];                                       
    PeopleLocations[i] = GetRandom(CELL_COUNT);                                 
    PeopleLocationsNew[i] = PeopleLocations[i];                                 
    PeopleInfectedTimeCounts[i] = 0;                                            
    CellStates[PeopleLocations[i] / CELL_COUNT_X]                               
              [PeopleLocations[i] % CELL_COUNT_X] = PeopleStates[i];            
  }                                                                             
}

