#include "move.h"

// Define a function to move people
void MovePeople() {
  for (int i = 0; i < PEOPLE_COUNT; i++) {
    switch (GetRandom(4)) {
      case 0:
        // If not on the top border, move up
        if (PeopleLocations[i] - CELL_COUNT_X >= 0) {
          PeopleLocationsNew[i] = PeopleLocations[i] - CELL_COUNT_X;
        }
        break;
      case 1:
        // If not on the left border, move left
        if (PeopleLocations[i] % CELL_COUNT_X - 1 >= 0) {
          PeopleLocationsNew[i] = PeopleLocations[i] - 1;
        }
        break;
      case 2:
        // If not on the bottom border, move down
        if (PeopleLocations[i] + CELL_COUNT_X < CELL_COUNT) {
          PeopleLocationsNew[i] = PeopleLocations[i] + CELL_COUNT_X;
        }
        break;
      default:
        // If not on the right border, move right
        if (PeopleLocations[i] % CELL_COUNT_X + 1 < CELL_COUNT_X) {
          PeopleLocationsNew[i] = PeopleLocations[i] + 1;
        }
        break;
    }
  }
}

