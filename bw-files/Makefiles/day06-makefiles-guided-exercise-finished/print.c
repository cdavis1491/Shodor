#include "print.h"

// Define a function to print the state of each cell in the environment
void PrintCells(int const time) {
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
}

