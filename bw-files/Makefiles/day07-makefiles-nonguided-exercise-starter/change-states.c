#include "change-states.h"

// Define a function to change peoples' states
void ChangePeopleStates() {
  for (int i = 0; i < PEOPLE_COUNT; i++) {
    PeopleStatesNew[i] = PeopleStates[i];

    // Infect susceptible people
    if (PeopleStates[i] == SUSCEPTIBLE) {
      for (int j = 0; j < PEOPLE_COUNT; j++) {
        // Don't compare a person to itself
        if (i == j) {
          continue;
        }
        if (PeopleStates[j] == INFECTED &&
            // If people are next to or on top of each other
            (PeopleLocations[i] == PeopleLocations[j]     ||
             PeopleLocations[i] == PeopleLocations[j] - 1 ||
             PeopleLocations[i] == PeopleLocations[j] + 1 ||
             PeopleLocations[i] == PeopleLocations[j] - CELL_COUNT_X ||
             PeopleLocations[i] == PeopleLocations[j] + CELL_COUNT_X) &&
            GetRandom(100) < CONTAGIOUSNESS) {
          PeopleStatesNew[i] = INFECTED;
          break;
        }
      }
    }

    // Recover infected people and advance their infected time counts
    else if (PeopleStates[i] == INFECTED) {
      PeopleInfectedTimeCounts[i]++;
      if (PeopleInfectedTimeCounts[i] == INFECTED_TIME_COUNT) {
        PeopleStatesNew[i] = RECOVERED;
      }
    }
  }
}

