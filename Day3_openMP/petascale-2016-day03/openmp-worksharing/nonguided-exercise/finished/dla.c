/* Diffusion Limited Aggregation
   Aaron Weeden, Shodor, 2016
 */

/*******************************************************************************
  IMPORT LIBRARIES
 ******************************************************************************/
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
#define PARTICLE_COUNT 201 // # of particles in the simulation
#define TIME_COUNT 200 // # of time steps in the simulation
#define ROW_COUNT 21  // Number of rows of cells in the environment
#define COLUMN_COUNT 50  // Number of columns of cells in the environment
#define COLUMN_COUNT_PLUS_NEWLINE ((COLUMN_COUNT)+1)
#define POSITION_COUNT ((ROW_COUNT) * (COLUMN_COUNT))
#define POSITION_COUNT_PLUS_NEWLINES ((ROW_COUNT) * (COLUMN_COUNT_PLUS_NEWLINE))
#define STICKINESS 50  // (0-100) likelihood of a particle sticking

// Define macros for particle states
#define NONE  ' '
#define FREE  'F'
#define STUCK 'S'

// Define macros movement directions
#define UP    0
#define LEFT  1
#define RIGHT 2
#define DOWN  3

#define OMP_NUM_THREADS 32

/*******************************************************************************
  DEFINE CUSTOM DATATYPES
 ******************************************************************************/
typedef struct
{
  int row;
  int column;
} Position_t;

typedef struct
{
  Position_t position;
  Position_t newPosition;
  char state;
  char newState;
} Particle_t;

/*******************************************************************************
  DECLARE GLOBAL VARIABLES
 ******************************************************************************/
Particle_t Particles[PARTICLE_COUNT]; // array of particles
char OutputStr[POSITION_COUNT_PLUS_NEWLINES]; // string for outputting the state
                                              // of each cell in the environment
int TimeIdx; // The current time step

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void SetOutputStrForParticle(int const particleIdx);
void Init();
  void InitRandomSeed();
  void InitParticles();
  void InitOutputStrNewlines();
void Simulate();
  void SetOutputStr();
  void SetNewPositions();
    void ChooseNewPositions();
    void DecideNewPositions();
  void SetNewStates();
  void PrintOutputStr();
  void AdvanceParticles();

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
int main()
{
  // If there are not enough empty spaces, there is an error; exit early
  if (PARTICLE_COUNT > POSITION_COUNT)
  {
    fprintf(stderr, "ERROR: There are not enough empty spaces; use a bigger row"
                    " count and/or column count or fewer particles\n");
    exit(EXIT_FAILURE);
  }

  // Set the number of OpenMP threads
  omp_set_num_threads(OMP_NUM_THREADS);

  // Enable nested OpenMP regions
  omp_set_nested(true);

  #ifdef DEBUG
    int threadCounter = 0;
    #pragma omp parallel reduction(+:threadCounter)
    {
      #pragma omp parallel reduction(+:threadCounter)
      {
        threadCounter++;
      }
    }
    printf("DEBUG: threadCounter: %d\n", threadCounter);
  #endif

  Init();
  Simulate();
  return 0;
}

// Helper function
void SetOutputStrForParticle(int const particleIdx)
{
  // Create some variables to help with readability
  Particle_t const particle = Particles[particleIdx];
  Position_t const position = particle.position;
  int        const row      = position.row;
  int        const column   = position.column;
  char       const state    = particle.state;

  OutputStr[row * COLUMN_COUNT_PLUS_NEWLINE + column] = state;
}

// Preconditions: none
// Postconditions: The random number generator has been seeded
//                 Particles has been initialized
//                 OutputStr has been initialized
void Init()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section
    {
      InitRandomSeed();
      InitParticles();
    }
    #pragma omp section
    {
      InitOutputStrNewlines();
    }
  }
}

// Preconditions: none
// Postconditions: The random number generator has been seeded
void InitRandomSeed()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #ifdef RANDOM_SEED
    srandom(RANDOM_SEED);
  #else
    srandom(time(NULL));
  #endif
}

// Preconditions: The random number generator has been seeded
// Postconditions: Particles has been initialized
//                 OutputStr has been initialized, except possibly for newlines
void InitParticles()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  // Create some variables to help readability
  Particle_t * particle_p    = &(Particles[0]);
  Position_t * position_p    = &(particle_p->position);
  Position_t * newPosition_p = &(particle_p->newPosition);

  // Initialize row, column, and state
  position_p->row    = newPosition_p->row    = 0.5 * ROW_COUNT    - 1;
  position_p->column = newPosition_p->column = 0.5 * COLUMN_COUNT - 1;
  particle_p->state  = particle_p->newState  = STUCK;

  // Initialize OutputStr
  SetOutputStrForParticle(0);

  // Initialize a list of open positions
  int openPositionIndices[POSITION_COUNT];

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
  for (int positionIdx = 0; positionIdx < POSITION_COUNT; positionIdx++)
  {
    #ifdef DEBUG
      DebugLoop(positionIdx, __FUNCTION__);
    #endif
 
    openPositionIndices[positionIdx] = positionIdx;
  }
 
  // Indicate the position for the stuck particle is not open
  openPositionIndices[position_p->row * COLUMN_COUNT +
                      position_p->column] = -1;
 
  // Initialize a count of open positions
  int openPositionCount = POSITION_COUNT-1;

  // This loop cannot be parallelized because openPositionIndices is being
  //   indexed randomly
  for (int particleIdx = 1; particleIdx < PARTICLE_COUNT; particleIdx++)
  {
    #ifdef DEBUG
      DebugLoop(particleIdx, __FUNCTION__);
    #endif

    // Find an open position
    int positionIdx = GetRandom(openPositionCount);
    while (openPositionIndices[positionIdx] == -1)
    {
      positionIdx++;
    }
    int const row    = openPositionIndices[positionIdx] / COLUMN_COUNT;
    int const column = openPositionIndices[positionIdx] % COLUMN_COUNT;
    openPositionIndices[positionIdx] = -1;
    openPositionCount--;

    // Create some variables to help readability
    particle_p    = &(Particles[particleIdx]);
    position_p    = &(particle_p->position);
    newPosition_p = &(particle_p->newPosition);

    // Initialize row, column, and state
    position_p->row    = newPosition_p->row    = row;
    position_p->column = newPosition_p->column = column;
    particle_p->state  = particle_p->newState  = FREE;

    // Initialize OutputStr
    SetOutputStrForParticle(particleIdx);
  }
}

// Preconditions: none
// Postconditions: OutputStr has been initialized for newlines
void InitOutputStrNewlines()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/2)
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    #ifdef DEBUG
      DebugLoop(rowIdx, __FUNCTION__);
    #endif

    OutputStr[(rowIdx+1) * COLUMN_COUNT_PLUS_NEWLINE - 1] = '\n';
  }
}

// Preconditions: The random number generator has been seeded
//                Particles has been initialized
//                OutputStr has been initialized
// Postconditions: The simulation has run
void Simulate()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  // This loop cannot be parallelized because each time step depends on the
  //   previous time step
  // Loop iteration start postconditions: Particles has not been updated at
  //                                        TimeIdx
  //                                      OutputStr has not been updated at
  //                                        TimeIdx
  // Loop iteration end preconditions: Particles has been updated at TimeIdx
  //                                   OutputStr has been updated at TimeIdx
  //                                   OutputStr has been printed at TimeIdx
  for (TimeIdx = 0; TimeIdx < TIME_COUNT; TimeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(TimeIdx, __FUNCTION__);
    #endif

    #pragma omp parallel sections num_threads(3)
    {
      #pragma omp section
      {
        SetOutputStr();
      }
      #pragma omp section
      {
        SetNewPositions();
      }
      #pragma omp section
      {
        SetNewStates();
      }
    }
    #pragma omp parallel sections num_threads(2)
    {
      #pragma omp section
      {
        PrintOutputStr();
      }
      #pragma omp section
      {
        AdvanceParticles();
      }
    }
  }
}

// Preconditions: Particles has not been updated at TimeIdx
// Postconditions: OutputStr has been updated at TimeIdx
void SetOutputStr()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/3)
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, rowIdx, __FUNCTION__);
    #endif

    for (int columnIdx = 0; columnIdx < COLUMN_COUNT; columnIdx++)
    {
      #ifdef DEBUG
        DebugDoublyNestedLoop(TimeIdx, rowIdx, columnIdx, __FUNCTION__);
      #endif

      OutputStr[rowIdx * COLUMN_COUNT_PLUS_NEWLINE + columnIdx] = NONE;
    }
  }

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/3)
  for (int particleIdx = 0; particleIdx < PARTICLE_COUNT; particleIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, particleIdx, __FUNCTION__);
    #endif

    SetOutputStrForParticle(particleIdx);
  }
}

// Preconditions: Particles[*].position has not been updated at TimeIdx
// Postconditions: Particles[*].newPosition has been updated at TimeIdx
void SetNewPositions()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  ChooseNewPositions();
  DecideNewPositions();
}

// Preconditions: Particles[*].newPosition has not been decided for TimeIdx
// Postconditions: Particles[*].newPosition has been chosen for TimeIdx
void ChooseNewPositions()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  #pragma omp parallel for num_threads(OMP_NUM_THREADS/3)
  for (int particleIdx = 0; particleIdx < PARTICLE_COUNT; particleIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, particleIdx, __FUNCTION__);
    #endif

    // Create some variables to help with readability
    Position_t   const position   = Particles[particleIdx].position;
    int          const row        = position.row;
    int          const column     = position.column;

    bool isPositionOpen[4] = { true, // up
                               true, // left
                               true, // down
                               true }; // right

    int openPositionCount = 4;

    // Check for wall collisions
    if (row == 0)
    {
      isPositionOpen[UP] = false;
      openPositionCount--;
    }
    if (column == 0)
    {
      isPositionOpen[LEFT] = false;
      openPositionCount--;
    }
    if (row == ROW_COUNT-1)
    {
      isPositionOpen[DOWN] = false;
      openPositionCount--;
    }
    if (column == COLUMN_COUNT-1)
    {
      isPositionOpen[RIGHT] = false;
      openPositionCount--;
    }

    // Check for collisions with other particles
    for (int otherParticleIdx = 0; openPositionCount > 0 &&
         otherParticleIdx < PARTICLE_COUNT; otherParticleIdx++)
    {
      #ifdef DEBUG
        DebugDoublyNestedLoop(TimeIdx, particleIdx, otherParticleIdx,
                              __FUNCTION__);
      #endif

      // Create some variables to help with readability
      Position_t const otherPosition = Particles[otherParticleIdx].position;
      int        const otherRow      = otherPosition.row;
      int        const otherColumn   = otherPosition.column;

      if (isPositionOpen[UP] && row == otherRow + 1 && column == otherColumn)
      {
        isPositionOpen[UP] = false;
        openPositionCount--;
      }
      else if (isPositionOpen[LEFT] && row == otherRow &&
               column == otherColumn + 1)
      {
        isPositionOpen[LEFT] = false;
        openPositionCount--;
      }
      else if (isPositionOpen[DOWN] && row == otherRow - 1 &&
               column == otherColumn)
      {
        isPositionOpen[DOWN] = false;
        openPositionCount--;
      }
      else if (isPositionOpen[RIGHT] && row == otherRow &&
               column == otherColumn - 1)
      {
        isPositionOpen[RIGHT] = false;
        openPositionCount--;
      }
    }

    // If there are no open positions, skip this particle
    if (openPositionCount == 0)
    {
      continue;
    }

    // Pick a random position
    int positionIdx = GetRandom(openPositionCount);
    // Loop until the position index points to an open position
    while (!isPositionOpen[positionIdx])
    {
      positionIdx++;
    }

    // Set the new position temporarily; it may be changed in
    //   DecideNewPositions
    Position_t * const newPosition_p = &(Particles[particleIdx].newPosition);
    if (positionIdx == UP)
    {
      newPosition_p->row    = row - 1;
      newPosition_p->column = column;
    }
    else if (positionIdx == LEFT)
    {
      newPosition_p->row    = row;
      newPosition_p->column = column - 1;
    }
    else if (positionIdx == DOWN)
    {
      newPosition_p->row    = row + 1;
      newPosition_p->column = column;
    }
    else if (positionIdx == RIGHT)
    {
      newPosition_p->row    = row;
      newPosition_p->column = column + 1;
    }
  }
}

// Preconditions: Particles[*].newPosition has been chosen for TimeIdx
// Postconditions: Particles[*].newPosition has been decided for TimeIdx
void DecideNewPositions()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  bool isDone[PARTICLE_COUNT];
  // Loop over each particle, mark it as not done
  #pragma omp parallel for num_threads(OMP_NUM_THREADS/3)
  for (int particleIdx = 0; particleIdx < PARTICLE_COUNT; particleIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, particleIdx, __FUNCTION__);
    #endif
 
    isDone[particleIdx] = false;
  }

  // Loop over each particle
  // This loop should not be parallelized because it relies on particles being
  //   processed one at a time
  for (int particleIdx = 0; particleIdx < PARTICLE_COUNT; particleIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, particleIdx, __FUNCTION__);
    #endif

    // If the particle is already done, skip it
    if (isDone[particleIdx])
    {
      continue;
    }

    // Create some variables to help readability
    Particle_t const particle    = Particles[particleIdx];
    Position_t const newPosition = particle.newPosition;
    int        const row         = newPosition.row;
    int        const column      = newPosition.column;

    // Create a list of particles to choose from to 
    int particleCandidateIndices[4];
    int particleCandidateCount = 0;
    particleCandidateIndices[particleCandidateCount++] = particleIdx;

    // Loop over each other particle
    // This loop should not be parallelized because particleCandidateIndices
    //   needs to be indexed in order
    for (int otherParticleIdx = particleIdx+1;
         otherParticleIdx < PARTICLE_COUNT; otherParticleIdx++)
    {
      #ifdef DEBUG
        DebugDoublyNestedLoop(TimeIdx, particleIdx, otherParticleIdx,
                              __FUNCTION__);
      #endif

      // If the other particle is already done, skip it
      if (isDone[otherParticleIdx])
      {
        continue;
      }

      // Create some variables to help readability
      Particle_t * const otherParticle_p  = &(Particles[otherParticleIdx]);
      Position_t * const otherNewPosition = &(otherParticle_p->newPosition);
      int          const otherRow         = otherNewPosition->row;
      int          const otherColumn      = otherNewPosition->column;

      // If both particles will be in the same position, add the other particle
      //  to the list of particle candidate
      if (row == otherRow && column == otherColumn)
      {
        particleCandidateIndices[particleCandidateCount++] = otherParticleIdx;
      }
    }

    // Choose a random particle to be the one to move into the open position
    int const choiceIdx = GetRandom(particleCandidateCount);
 
    // For each of the candidate particles,
    // We won't need more threads than particleCandidateCount, which is max 4
    #pragma omp parallel for num_threads(particleCandidateCount)
    for (int particleCandidateIdx = 0;
        particleCandidateIdx < particleCandidateCount; particleCandidateIdx++)
    {
      int const idx = particleCandidateIndices[particleCandidateIdx];
  
      // If the particle was not chosen, tell it to stay where it is
      if (particleCandidateIdx != choiceIdx)
      {
        Particle_t * const particle_p    = &(Particles[idx]);
        Position_t * const position_p    = &(particle_p->position);
        Position_t * const newPosition_p = &(particle_p->newPosition);
        newPosition_p->row    = position_p->row;
        newPosition_p->column = position_p->column;
      }
 
      // Set the particle as done
      isDone[idx] = true;
    }
  }
}

// Preconditions: Particles[*].state has not been updated at TimeIdx
// Postconditions: Particles[*].newState has been updated at TimeIdx
void SetNewStates()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  // Loop over each particle
  #pragma omp parallel for num_threads(OMP_NUM_THREADS/3)
  for (int particleIdx = 0; particleIdx < PARTICLE_COUNT; particleIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, particleIdx, __FUNCTION__);
    #endif

    // Create variables to help readability
    Particle_t * const particle_p = &(Particles[particleIdx]);
    Position_t   const position   = particle_p->position;
    int          const row        = position.row;
    int          const column     = position.column;

    // Loop over each other particle
    for (int otherParticleIdx = 0; otherParticleIdx < PARTICLE_COUNT;
         otherParticleIdx++)
    {
      #ifdef DEBUG
        DebugDoublyNestedLoop(TimeIdx, particleIdx, otherParticleIdx,
                              __FUNCTION__);
      #endif

      // If the two particles are the same, skip
      if (particleIdx == otherParticleIdx)
      {
        continue;
      }

      // Create variables to help readability
      Particle_t const otherParticle = Particles[otherParticleIdx];

      // If the other particle is free, ignore it
      if (otherParticle.state == FREE)
      {
        continue;
      }

      // Create variables to help readability
      Position_t const otherPosition = otherParticle.position;
      int        const otherRow      = otherPosition.row;
      int        const otherColumn   = otherPosition.column;

      // If the two particles are next to each other, and with some percent
      //  chance,
      if (((row == otherRow && (column == otherColumn+1 ||
                               column == otherColumn-1)) ||
           (column == otherColumn && (row == otherRow+1 ||
                                      row == otherRow-1))) &&
          GetRandom(100) < STICKINESS)
      {
        // Stick the first particle
        particle_p->newState = STUCK;

        // We are done looking for other particles
        break;
      }
    }
  }
}

// Preconditions: OutputStr has been updated at TimeIdx
// Postconditions: OutputStr has been printed at TimeIdx
void PrintOutputStr()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  printf("Time step %d\n%s\n", TimeIdx, OutputStr);
}

// Preconditions: Particles[*].newPosition has been updated at TimeIdx
//                Particles[*].newState has been updated at TimeIdx
// Postconditions: Particles has been updated at TimeIdx
void AdvanceParticles()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  // Loop over all particles
  #pragma omp parallel for num_threads(OMP_NUM_THREADS)
  for (int particleIdx = 0; particleIdx < PARTICLE_COUNT; particleIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, particleIdx, __FUNCTION__);
    #endif

    // Create variables to help readability
    Particle_t * const particle_p    = &(Particles[particleIdx]);
    Position_t * const position_p    = &(particle_p->position);
    Position_t * const newPosition_p = &(particle_p->newPosition);
    int          const newState      = particle_p->newState;

    // Don't move stuck particles
    if (newState == STUCK)
    {
      newPosition_p->row    = position_p->row;
      newPosition_p->column = position_p->column;
    }

    // Advance
    particle_p->state  = newState;
    position_p->row    = newPosition_p->row;
    position_p->column = newPosition_p->column;
  }
}

