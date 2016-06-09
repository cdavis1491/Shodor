#include "update.h"

// Define a function to update the model to the next time step
void Update() {
  // Copy back the new people arrays
  CopyPeopleArrays();

  // Update the cells array
  UpdateCells();
}

// Define a function to copy back the new people arrays
void CopyPeopleArrays() {
  for (int i = 0; i < PEOPLE_COUNT; i++) {
    PeopleStates[i] = PeopleStatesNew[i];
    PeopleLocations[i] = PeopleLocationsNew[i];
  }
}

// Define a function to update the cells array
void UpdateCells() {
  for (int y = 0; y < CELL_COUNT_Y; y++) {
    for (int x = 0; x < CELL_COUNT_X; x++) {
      CellStates[y][x] = EMPTY;
    }
  }
  for (int i = 0; i < PEOPLE_COUNT; i++) {
    CellStates[PeopleLocations[i] / CELL_COUNT_X]
              [PeopleLocations[i] % CELL_COUNT_X] = PeopleStates[i];
  }
}

