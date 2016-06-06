// SIR model - A certain number of people (PERSON_COUNT) move randomly in a 2D
// world of cells (with dimensions ROW_COUNT by COLUMN_COUNT). Each cell can
// only be occupied by one person at a time. Some people start out infected;
// the rest start out susceptible. If an infected person is directly next to
// (above, to the left, to the right, or below) a susceptible
// person, the susceptible person has some percent chance of becoming infected
// (CONTAGIOUSNESS). After an infected person has been infected for a certain
// number of time steps (INFECTED_TIME_COUNT), the person becomes recovered.
// The model runs for a certain number of time steps (TIME_COUNT).
//
// Author: Aaron Weeden, Shodor, 2016

/*******************************************************************************
  IMPORT LIBRARIES
 ******************************************************************************/
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
#define PERSON_COUNT 50 // # of people in the simulation
#define TIME_COUNT 200 // # of time steps in the simulation
#define INIT_INFECTED_COUNT 1 // # of initially infected people in the
                              // simulation
#define ROW_COUNT 21 // Number of rows of cells in the environment
#define COLUMN_COUNT 50 // Number of columns of cells in the environment
#define COLUMN_COUNT_PLUS_NEWLINE ((COLUMN_COUNT)+1)
#define POSITION_COUNT ((ROW_COUNT) * (COLUMN_COUNT))
#define POSITION_COUNT_PLUS_NEWLINES ((ROW_COUNT) * (COLUMN_COUNT_PLUS_NEWLINE))
#define INFECTED_TIME_COUNT 50 // number of time steps a person stays infected
#define CONTAGIOUSNESS 30 // [0-100] likelihood of person catching the disease

// Define macros for people states
#define NONE        ' '
#define SUSCEPTIBLE 's'
#define INFECTED    'i'
#define RECOVERED   'r'

// Define macros movement directions
#define UP    0
#define LEFT  1
#define RIGHT 2
#define DOWN  3

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
  int infectedTimeCount;
} Person_t;

/*******************************************************************************
  DECLARE GLOBAL VARIABLES
 ******************************************************************************/
Person_t People[PERSON_COUNT]; // array of people
char OutputStr[POSITION_COUNT_PLUS_NEWLINES]; // string for outputting the state
                                              // of each cell in the environment
int TimeIdx; // The current time step

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void SetOutputStrForPerson(int const personIdx);
void Init();
  void InitRandomSeed();
  void InitPeople();
  void InitOutputStrNewlines();
void Simulate();
  void SetOutputStr();
  void SetNewPositions();
    void ChooseNewPositions();
    void DecideNewPositions();
  void SetNewStates();
    void InfectPeople();
    void RecoverPeople();
  void PrintOutputStr();
  void AdvancePeople();
void PrintFinalCounts();

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
int main() {
  // If there are not enough empty spaces, there is an error; exit early
  if (PERSON_COUNT > POSITION_COUNT)
  {
    fprintf(stderr, "ERROR: There are not enough empty spaces; use a bigger row"
                    " count and/or column count or fewer people\n");
    exit(EXIT_FAILURE);
  }

  Init();
  Simulate();
  PrintFinalCounts();
  return 0;
}

// Helper function
void SetOutputStrForPerson(int const personIdx)
{
  // Create some variables to help with readability
  Person_t   const person   = People[personIdx];
  Position_t const position = person.position;
  int        const row      = position.row;
  int        const column   = position.column;
  char       const state    = person.state;

  OutputStr[row * COLUMN_COUNT_PLUS_NEWLINE + column] = state;
}

// Preconditions: none
// Postconditions: The random number generator has been seeded
//                 People has been initialized
//                 OutputStr has been initialized
void Init()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  InitRandomSeed();
  InitPeople();
  InitOutputStrNewlines();
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
// Postconditions: People has been initialized
//                 OutputStr has been initialized, except possibly for newlines
void InitPeople()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  // Initialize a list of open positions
  int openPositionIndices[POSITION_COUNT];

  for (int positionIdx = 0; positionIdx < POSITION_COUNT; positionIdx++)
  {
    #ifdef DEBUG
      DebugLoop(positionIdx, __FUNCTION__);
    #endif

    openPositionIndices[positionIdx] = positionIdx;
  }

  // Initialize a count of open positions
  int openPositionCount = POSITION_COUNT-1;

  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugLoop(personIdx, __FUNCTION__);
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
    Person_t   * const person_p      = &(People[personIdx]);
    Position_t * const position_p    = &(person_p->position);
    Position_t * const newPosition_p = &(person_p->newPosition);

    // Initialize row, column, state, and infected time count
    position_p->row    = newPosition_p->row    = row;
    position_p->column = newPosition_p->column = column;
    int state = SUSCEPTIBLE;
    if (personIdx < INIT_INFECTED_COUNT)
    {
      state = INFECTED;
    }
    person_p->state = person_p->newState = state;
    person_p->infectedTimeCount = 0;

    // Initialize OutputStr
    SetOutputStrForPerson(personIdx);
  }
}

// Preconditions: none
// Postconditions: OutputStr has been initialized for newlines
void InitOutputStrNewlines()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    #ifdef DEBUG
      DebugLoop(rowIdx, __FUNCTION__);
    #endif

    OutputStr[(rowIdx+1) * COLUMN_COUNT_PLUS_NEWLINE - 1] = '\n';
  }
}

// Preconditions: The random number generator has been seeded
//                People has been initialized
//                OutputStr has been initialized
// Postconditions: The simulation has run
void Simulate()
{
  #ifdef DEBUG
    DebugFunction(__FUNCTION__);
  #endif

  // Loop iteration start postconditions: People has not been updated at
  //                                        TimeIdx
  //                                      OutputStr has not been updated at
  //                                        TimeIdx
  // Loop iteration end preconditions: People has been updated at TimeIdx
  //                                   OutputStr has been updated at TimeIdx
  //                                   OutputStr has been printed at TimeIdx
  for (TimeIdx = 0; TimeIdx < TIME_COUNT; TimeIdx++)
  {
    #ifdef DEBUG
      DebugLoop(TimeIdx, __FUNCTION__);
    #endif

    SetOutputStr();
    SetNewPositions();
    SetNewStates();
    PrintOutputStr();
    AdvancePeople();
  }
}

// Preconditions: People has not been updated at TimeIdx
// Postconditions: OutputStr has been updated at TimeIdx
void SetOutputStr()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

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

  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, personIdx, __FUNCTION__);
    #endif

    SetOutputStrForPerson(personIdx);
  }
}

// Preconditions: People[*].position has not been updated at TimeIdx
// Postconditions: People[*].newPosition has been updated at TimeIdx
void SetNewPositions()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  ChooseNewPositions();
  DecideNewPositions();
}

// Preconditions: People[*].newPosition has not been decided for TimeIdx
// Postconditions: People[*].newPosition has been chosen for TimeIdx
void ChooseNewPositions()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, personIdx, __FUNCTION__);
    #endif

    // Create some variables to help with readability
    Position_t   const position   = People[personIdx].position;
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

    // Check for collisions with other people
    for (int otherPersonIdx = 0; openPositionCount > 0 &&
         otherPersonIdx < PERSON_COUNT; otherPersonIdx++)
    {
      #ifdef DEBUG
        DebugDoublyNestedLoop(TimeIdx, personIdx, otherPersonIdx, __FUNCTION__);
      #endif

      // Create some variables to help with readability
      Position_t const otherPosition = People[otherPersonIdx].position;
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

    // If there are no open positions, skip this person
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
    Position_t * const newPosition_p = &(People[personIdx].newPosition);
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

// Preconditions: People[*].newPosition has been chosen for TimeIdx
// Postconditions: People[*].newPosition has been decided for TimeIdx
void DecideNewPositions()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  bool isDone[PERSON_COUNT];
  // Loop over each person, mark it as not done
  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, personIdx, __FUNCTION__);
    #endif

    isDone[personIdx] = false;
  }

  // Loop over each person
  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, personIdx, __FUNCTION__);
    #endif

    // If the person is already done, skip it
    if (isDone[personIdx])
    {
      continue;
    }

    // Create some variables to help readability
    Person_t const person    = People[personIdx];
    Position_t const newPosition = person.newPosition;
    int        const row         = newPosition.row;
    int        const column      = newPosition.column;

    // Create a list of people to choose from to 
    int personCandidateIndices[4];
    int personCandidateCount = 0;
    personCandidateIndices[personCandidateCount++] = personIdx;

    // Loop over each other person
    for (int otherPersonIdx = personIdx+1;
         otherPersonIdx < PERSON_COUNT; otherPersonIdx++)
    {
      #ifdef DEBUG
        DebugDoublyNestedLoop(TimeIdx, personIdx, otherPersonIdx,
                              __FUNCTION__);
      #endif

      // If the other person is already done, skip it
      if (isDone[otherPersonIdx])
      {
        continue;
      }

      // Create some variables to help readability
      Person_t   * const otherPerson_p  = &(People[otherPersonIdx]);
      Position_t * const otherNewPosition = &(otherPerson_p->newPosition);
      int          const otherRow         = otherNewPosition->row;
      int          const otherColumn      = otherNewPosition->column;

      // If both people will be in the same position, add the other person
      //  to the list of person candidate
      if (row == otherRow && column == otherColumn)
      {
        personCandidateIndices[personCandidateCount++] = otherPersonIdx;
      }
    }

    // Choose a random person to be the one to move into the open position
    int const choiceIdx = GetRandom(personCandidateCount);

    // For each of the candidate people,
    for (int personCandidateIdx = 0;
        personCandidateIdx < personCandidateCount; personCandidateIdx++)
    {
      int const idx = personCandidateIndices[personCandidateIdx];

      // If the person was not chosen, tell it to stay where it is
      if (personCandidateIdx != choiceIdx)
      {
        Person_t   * const person_p      = &(People[idx]);
        Position_t * const position_p    = &(person_p->position);
        Position_t * const newPosition_p = &(person_p->newPosition);
        newPosition_p->row    = position_p->row;
        newPosition_p->column = position_p->column;
      }

      // Set the person as done
      isDone[idx] = true;
    }
  }
}

// Preconditions: People[*].state has not been updated at TimeIdx
// Postconditions: People[*].newState has been updated at TimeIdx
void SetNewStates()
{
  InfectPeople();
  RecoverPeople();
}

// Preconditions: People[*].state has not been updated at TimeIdx
// Postconditions: People[*].newState has been updated at TimeIdx for some
//                   susceptible->infected people
void InfectPeople()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  // Loop over each person
  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, personIdx, __FUNCTION__);
    #endif

    // Create variables to help readability
    Person_t   * const person_p = &(People[personIdx]);
    Position_t   const position = person_p->position;
    int          const row      = position.row;
    int          const column   = position.column;

    // Loop over each other person
    for (int otherPersonIdx = 0; otherPersonIdx < PERSON_COUNT;
         otherPersonIdx++)
    {
      #ifdef DEBUG
        DebugDoublyNestedLoop(TimeIdx, personIdx, otherPersonIdx,
                              __FUNCTION__);
      #endif

      // If the two people are the same, skip
      if (personIdx == otherPersonIdx)
      {
        continue;
      }

      // Create variables to help readability
      Person_t const otherPerson = People[otherPersonIdx];

      // If the other person is not infected, skip
      if (otherPerson.state != INFECTED)
      {
        continue;
      }

      // Create variables to help readability
      Position_t const otherPosition = otherPerson.position;
      int        const otherRow      = otherPosition.row;
      int        const otherColumn   = otherPosition.column;

      // If the two people are next to each other, and with some percent
      //  chance,
      if (((row == otherRow && (column == otherColumn+1 ||
                                column == otherColumn-1)) ||
           (column == otherColumn && (row == otherRow+1 ||
                                      row == otherRow-1))) &&
          GetRandom(100) < CONTAGIOUSNESS)
      {
        // Make the first person infected
        person_p->newState = INFECTED;

        // We are done looking for other people
        break;
      }
    }
  }
}

// Preconditions: People[*].state has not been updated at TimeIdx
//                People[*].infectedTimeCount has not been updated at TimeIdx
// Postconditions: People[*].newState has been updated at TimeIdx for some
//                   infected->recovered people
void RecoverPeople()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, personIdx, __FUNCTION__);
    #endif

    Person_t * const person_p = &(People[personIdx]);

    if (person_p->state == INFECTED &&
        person_p->infectedTimeCount == INFECTED_TIME_COUNT)
    {
      person_p->newState = RECOVERED;
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

// Preconditions: People[*].newPosition has been updated at TimeIdx
//                People[*].newState has been updated at TimeIdx
// Postconditions: People has been updated at TimeIdx
void AdvancePeople()
{
  #ifdef DEBUG
    DebugFunctionInLoop(TimeIdx, __FUNCTION__);
  #endif

  // Loop over all people
  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    #ifdef DEBUG
      DebugNestedLoop(TimeIdx, personIdx, __FUNCTION__);
    #endif

    // Create variables to help readability
    Person_t   * const person_p      = &(People[personIdx]);
    Position_t * const position_p    = &(person_p->position);
    Position_t * const newPosition_p = &(person_p->newPosition);
    int          const newState      = person_p->newState;

    // Advance
    if (person_p->state == INFECTED)
    {
      person_p->infectedTimeCount++;
    }
    person_p->state    = newState;
    position_p->row    = newPosition_p->row;
    position_p->column = newPosition_p->column;
  }
}

// Preconditions: The simulation has run
// Postconditions: The final number of susceptible, infected, and recovered
//                   people has been printed
void PrintFinalCounts()
{
  int susceptibleCount = 0;
  int infectedCount = 0;
  int recoveredCount = 0;
  for (int personIdx = 0; personIdx < PERSON_COUNT; personIdx++)
  {
    switch (People[personIdx].state)
    {
      case SUSCEPTIBLE:
        susceptibleCount++;
        break;
      case INFECTED:
        infectedCount++;
        break;
      case RECOVERED:
        recoveredCount++;
        break;
    }
  }
  printf("Final counts: %d susceptible, %d infected, %d recovered\n",
         susceptibleCount, infectedCount, recoveredCount);
}

