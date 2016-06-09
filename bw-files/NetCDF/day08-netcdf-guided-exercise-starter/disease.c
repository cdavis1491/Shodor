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
#include <stdio.h>
#include <netcdf.h>
#include <stdlib.h>
#include <string.h>

// Define macros
#define PEOPLE_COUNT           50
#define INITIAL_INFECTED_COUNT 1
#define CELL_COUNT_X           10
#define CELL_COUNT_Y           21
#define CELL_COUNT             ((CELL_COUNT_X) * (CELL_COUNT_Y))
#define TIME_COUNT             200
#define INFECTED_TIME_COUNT    50
#define CONTAGIOUSNESS         30
#define RANDOM_SEED            0
#define SUSCEPTIBLE            0
#define INFECTED               1
#define RECOVERED              2
#define EMPTY                  3

// Declare the global arrays of people and cell data
int CellStates[CELL_COUNT_Y][CELL_COUNT_X];
int PeopleStates[PEOPLE_COUNT];
int PeopleStatesNew[PEOPLE_COUNT];
int PeopleLocations[PEOPLE_COUNT];
int PeopleLocationsNew[PEOPLE_COUNT];
int PeopleInfectedTimeCounts[PEOPLE_COUNT];

// Define a function that returns a random whole number
int GetRandom(int const max) {
  return max * random() / RAND_MAX;
}

// Define the main function
int main() {
  // Seed the random number generator
  srandom(RANDOM_SEED);

  // Initialize the array of cell data
  for (int y = 0; y < CELL_COUNT_Y; y++) {
    for (int x = 0; x < CELL_COUNT_X; x++) {
      CellStates[y][x] = EMPTY;
    }
  }

  // Initialize the arrays of people data and the cells where people are
  for (int i = 0; i < PEOPLE_COUNT; i++) {
    PeopleStates[i] = (i < INITIAL_INFECTED_COUNT) ? INFECTED : SUSCEPTIBLE;
    PeopleStatesNew[i] = PeopleStates[i];
    PeopleLocations[i] = GetRandom(CELL_COUNT);
    PeopleLocationsNew[i] = PeopleLocations[i];
    PeopleInfectedTimeCounts[i] = 0;
    CellStates[PeopleLocations[i] / CELL_COUNT_X]
              [PeopleLocations[i] % CELL_COUNT_X] = PeopleStates[i];
  }

  // Start the simulation
  for (int time = 0; time < TIME_COUNT; time++) {
    // Print the state of each cell in the environment
    printf("Time: %d\n", time);
    for (int y = CELL_COUNT_Y - 1; y >= 0; y--) {
      for (int x = 0; x < CELL_COUNT_X; x++) {

        // Print the given cell's state
        switch (CellStates[y][x]) {
          case SUSCEPTIBLE:
            printf("S");
            break;
          case INFECTED:
            printf("I");
            break;
          case RECOVERED:
            printf("R");
            break;
          default:
            printf(" ");
        }
      }
      // Print a newline if we are in the last cell in the row
      printf("\n");
    }
    // Print a newline in between time steps
    printf("\n");

    // Move people
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

    // Change peoples' states
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
  
    // Copy back the new arrays
    for (int i = 0; i < PEOPLE_COUNT; i++) {
      PeopleStates[i] = PeopleStatesNew[i];
      PeopleLocations[i] = PeopleLocationsNew[i];
    }

    // Update the cells array
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

  // Exit the program
  return 0;
}

