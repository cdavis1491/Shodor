/* Diffusion Limited Aggregation with NetCDF
 * Aaron Weeden, Shodor, 2016
 */

// Import libraries
#include <netcdf.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Set model parameters
#define PARTICLE_COUNT     201  // # of particles in the simulation
#define TIME_COUNT         50   // # of time steps in the simulation
#define ENVIRONMENT_LENGTH 50  // Length of one side of the square environment,
                               // measured in particle lengths
#define STICKINESS         100  // (0-100) likelihood of a particle sticking

// Define constants for particle states
#define NONE  0
#define FREE  1
#define STUCK 2

// Define a type for movement directions
typedef enum {
  UP, LEFT, RIGHT, DOWN
} Direction;

// Declare global arrays to hold the particle locations and states
int ParticleXs[PARTICLE_COUNT];
int ParticleYs[PARTICLE_COUNT];
int ParticleStates[PARTICLE_COUNT];
int Environment[ENVIRONMENT_LENGTH][ENVIRONMENT_LENGTH];

// Define NetCDF macros
#define FILENAME "dla.nc"
#define INDEPENDENT_VAR_COUNT 3 // time, y, x
#define TIME_NAME "Time"
#define TIME_UNITS "time steps since the simulation started"
#define Y_NAME "Y"
#define Y_UNITS "particle lengths from the bottom"
#define X_NAME "X"
#define X_UNITS "particle lengths from the left"
#define STATE_NAME "State"

// Declare NetCDF ID
int NcId;

// Declare NetCDF variable IDs
int TimeVarId;
int YVarId;
int XVarId;
int StateVarId;

// Declare functions
int GetRandom(int const max);
void TryNc(int const err);
void NcInit();
  void PutDimVals(int const varId, int const count);
void Init();
void NcOutput(int const time);
void StickParticles();
void MoveRandom();
  bool wouldCollide(int const i, Direction const dir);
void NcFinalize();

// Define the main function
int main() {
  // Seed the random number generator
  srandom(0);

  NcInit();
  Init();

  for (int time = 0; time < TIME_COUNT; time++) {
    NcOutput(time);
    StickParticles();
    MoveRandom();
  }

  NcFinalize();
  return 0;
}

// Define a function that returns a random integer less than a given maximum
int GetRandom(int const max) {
  return max * random() / RAND_MAX; 
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
  TryNc(nc_def_dim(NcId, TIME_NAME, TIME_COUNT,         &(dimIds[0])));
  TryNc(nc_def_dim(NcId, Y_NAME,    ENVIRONMENT_LENGTH, &(dimIds[1])));
  TryNc(nc_def_dim(NcId, X_NAME,    ENVIRONMENT_LENGTH, &(dimIds[2])));

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
  PutDimVals(YVarId, ENVIRONMENT_LENGTH);
  PutDimVals(XVarId, ENVIRONMENT_LENGTH);
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

// Define a function for initializing particle locations and states
void Init() {
  // Set the environment as having no state
  for (int y = 0; y < ENVIRONMENT_LENGTH; y++) {
    for (int x = 0; x < ENVIRONMENT_LENGTH; x++) {
      Environment[y][x] = NONE;
    }
  }

  // Center and stick the first particle
  ParticleXs[0] = ParticleYs[0] = ENVIRONMENT_LENGTH / 2 - 1;
  ParticleStates[0] = STUCK;
  Environment[ParticleYs[0]][ParticleXs[0]] = STUCK;

  // Loop over the other particles, give them random positions, and make them
  // free
  for (int i = 1; i < PARTICLE_COUNT; i++) {
    ParticleXs[i] = GetRandom(ENVIRONMENT_LENGTH);
    ParticleYs[i] = GetRandom(ENVIRONMENT_LENGTH);
    ParticleStates[i] = FREE;
    Environment[ParticleYs[i]][ParticleXs[i]] = FREE;
  }
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
    ENVIRONMENT_LENGTH, ENVIRONMENT_LENGTH // Write all positions in the
                                           // environment
  };

  // Output the C array into the NetCDF variable
  TryNc(nc_put_vara_int(NcId, // NetCDF dataset ID
                         StateVarId, // NetCDF variable ID
                         startArray, // Where to start writing data in each
                                     // dimension
                         countArray, // How much data to write in each
                                     // dimension
                         &(Environment[0][0]))); // C array of data to output to
                                                 // the NetCDF variable
}

// Define a function that sticks free particles
void StickParticles() {
  // Loop over all particles
  for (int i = 0; i < PARTICLE_COUNT; i++) {

    // If the particle is stuck, ignore it
    if (ParticleStates[i] == STUCK) {
      continue;
    }

    // Loop over all the other particles
    for (int j = 0; j < PARTICLE_COUNT; j++) {

      // If the particle is free, ignore it
      if (ParticleStates[j] == FREE) {
        continue;
      }

      // If the two particles are not adjacent, ignore them
      if (ParticleXs[i] > ParticleXs[j] + 1 ||
          ParticleXs[j] > ParticleXs[i] + 1 ||
          ParticleYs[i] > ParticleYs[j] + 1 ||
          ParticleYs[j] > ParticleYs[i] + 1) {
        continue;
      }

      // Stick the particle with some % chance
      if (GetRandom(100) < STICKINESS) {
        ParticleStates[i] = STUCK;
        Environment[ParticleYs[i]][ParticleXs[i]] = STUCK;
      }
    }
  }
}

// Define a function that moves free particles randomly
void MoveRandom() {
  // Loop over all particles
  for (int i = 0; i < PARTICLE_COUNT; i++) {
    // If the particle is stuck, skip it
    if (ParticleStates[i] == STUCK) {
      continue;
    }

    // Get random direction
    Direction dir;
    switch (GetRandom(4)) {
      case 0:
        dir = UP;
        break;
      case 1:
        dir = LEFT;
        break;
      case 2:
        dir = RIGHT;
        break;
      default:
        dir = DOWN;
        break;
    }

    // If the move would cause a collision with a wall, skip this particle
    if ((dir == UP    && ParticleYs[i] <= 0) ||
        (dir == LEFT  && ParticleXs[i] <= 0) ||
        (dir == RIGHT && ParticleXs[i] >= ENVIRONMENT_LENGTH - 1) ||
        (dir == DOWN  && ParticleYs[i] >= ENVIRONMENT_LENGTH - 1)) {
      continue;
    }

    // If the move would cause a collision with another particle, skip this
    // particle
    if (wouldCollide(i, dir)) {
      continue;
    }

    // Commit the move
    Environment[ParticleYs[i]][ParticleXs[i]] = NONE;
    switch (dir) {
      case UP:
        ParticleYs[i]--;
        break;
      case LEFT:
        ParticleXs[i]--;
        break;
      case RIGHT:
        ParticleXs[i]++;
        break;
      case DOWN:
        ParticleYs[i]++;
        break;
    }
    Environment[ParticleYs[i]][ParticleXs[i]] = FREE;
  }
}

// Define a function that returns whether a particle, moving in a given
// direction, would collide with another particle
bool wouldCollide(int const i, Direction const dir) {
  for (int j = 0; j < PARTICLE_COUNT; j++) {
    // Do not check for a particle colliding with itself
    if (i == j) {
      continue;
    }

    if ((dir == UP &&
         ParticleXs[j] == ParticleXs[i] &&
         ParticleYs[j] == ParticleYs[i] - 1) ||
        (dir == LEFT &&
         ParticleYs[j] == ParticleYs[i] &&
         ParticleXs[j] == ParticleXs[i] - 1) ||
        (dir == RIGHT &&
         ParticleYs[j] == ParticleYs[i] &&
         ParticleXs[j] == ParticleXs[i] + 1) ||
        (dir == DOWN &&
         ParticleXs[j] == ParticleXs[i] &&
         ParticleYs[j] == ParticleYs[i] + 1)) {
      return true;
    }
  }
  return false;
}

// Close the NetCDF dataset
void NcFinalize() {
  TryNc(nc_close(NcId));
}

