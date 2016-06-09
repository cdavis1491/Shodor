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
#include <netcdf.h>
#include <stdio.h>
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

// Define NetCDF macros
#define FILENAME "disease.nc"
#define INDEPENDENT_VAR_COUNT 3 // time, y, x
#define TIME_NAME "Time"
#define TIME_UNITS "time steps since the simulation started"
#define Y_NAME "Y"
#define Y_UNITS "person lengths from the bottom"
#define X_NAME "X"
#define X_UNITS "person lengths from the left"
#define STATE_NAME "State"

// Declare NetCDF ID
int NcId;

// Declare NetCDF variable IDs
int TimeVarId;
int YVarId;
int XVarId;
int StateVarId;

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

// Declare NetCDF functions
void TryNc(int const err);
void NcInit();
  void PutDimVals(int const varId, int const count);
void NcOutput(int const time);
void NcFinalize();

// Define the main function
int main() {
  NcInit();

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
    NcOutput(time);
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

  NcFinalize();

  // Exit the program
  return 0;
}

// Define a function that checks the return of a NetCDF function and, if there
// is an error, prints the error and exits
void TryNc(int const err) {
  if (err) {
    fprintf(stderr, "%s\n", nc_strerror(err));
    exit(EXIT_FAILURE);
  }
}

// Define a function for initializing NetCDF
void NcInit() {
  // Create the NetCDF dataset
  TryNc(nc_create(FILENAME, NC_CLOBBER, &NcId));

  // Declare dimension IDs for each independent variable
  int dimIds[INDEPENDENT_VAR_COUNT];

  // Define NetCDF dimensions for each independent variable
  TryNc(nc_def_dim(NcId, TIME_NAME, TIME_COUNT,   &(dimIds[0])));
  TryNc(nc_def_dim(NcId, Y_NAME,    CELL_COUNT_Y, &(dimIds[1])));
  TryNc(nc_def_dim(NcId, X_NAME,    CELL_COUNT_X, &(dimIds[2])));

  // Define NetCDF variables for each variable
  TryNc(nc_def_var(NcId, TIME_NAME,  NC_INT, 1, &(dimIds[0]), &TimeVarId));
  TryNc(nc_def_var(NcId, Y_NAME,     NC_INT, 1, &(dimIds[1]), &YVarId));
  TryNc(nc_def_var(NcId, X_NAME,     NC_INT, 1, &(dimIds[2]), &XVarId));
  TryNc(nc_def_var(NcId, STATE_NAME, NC_INT, INDEPENDENT_VAR_COUNT, dimIds,
                    &StateVarId));

  // Define NetCDF units text for each variable
  TryNc(nc_put_att_text(NcId, TimeVarId,  "units", strlen(TIME_UNITS),
                         TIME_UNITS));
  TryNc(nc_put_att_text(NcId, YVarId,     "units", strlen(Y_UNITS),
                         Y_UNITS));
  TryNc(nc_put_att_text(NcId, XVarId,     "units", strlen(X_UNITS),
                         X_UNITS));

  // Leave NetCDF define mode
  TryNc(nc_enddef(NcId));

  // Write the values for the each dimension
  PutDimVals(TimeVarId, TIME_COUNT);
  PutDimVals(YVarId, CELL_COUNT_Y);
  PutDimVals(XVarId, CELL_COUNT_X);
}

// Define a function that outputs values for a NetCDF dimension's variable
void PutDimVals(int const varId, int const count) {
  // Fill a 1D array with the numbers 0 through count
  int vals[count];
  for (int i = 0; i < count; i++) {
    vals[i] = i;
  }

  // Write the array to the given NetCDF variable
  TryNc(nc_put_var_int(NcId, varId, vals));
}

// Close the NetCDF dataset
void NcFinalize() {
  TryNc(nc_close(NcId));
}

// Define function to output NetCDF dependent variables
void NcOutput(int const time) {
  // Declare where to start writing data for each dimension
  size_t const startArray[INDEPENDENT_VAR_COUNT] = {
    time, // Start at the current time step
    0, 0 // Start at coordinate (0, 0)
  };

  // Declare how much data to write in each dimension
  size_t const countArray[INDEPENDENT_VAR_COUNT] = {
    1, // Write one time step of data
    CELL_COUNT_Y, CELL_COUNT_X // Write all positions in the environment
  };

  // Output the C array into the NetCDF variable
  TryNc(nc_put_vara_int(NcId, // NetCDF dataset ID
                         StateVarId, // NetCDF variable ID
                         startArray, // Where to start writing data in each
                                     // dimension
                         countArray, // How much data to write in each
                                     // dimension
                         &(CellStates[0][0]))); // C array of data to output to
                                                // the NetCDF variable
}

