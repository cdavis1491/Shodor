#include "utils.h"

/* Define a mapping from the row and column of a given tree in a forest with
   boundaries to the index of that tree in a 1D array that includes
   boundaries */
#define TREE_MAP(row, col, nColsPlusBounds) ((row) * (nColsPlusBounds) + (col))

/* Define a mapping from the row and column of a given tree in a forest with
   boundaries to the index of that tree in a 1D array that does not include
   boundaries */
#define NEW_TREE_MAP(row, col, nCols) ((row - 1) * (nCols) + (col - 1))

/* Generate a random integer between [min..max)

   @param min Smallest integer to generate
   @param max 1 more than the biggest integer to generate
   @return random integer
   */
int RandBetween(const int min, const int max) {
  return min + (random() % (max - min));
}

/* Return whether a given tree has burnt out

   @param row The row index of the tree 
   @param col The column index of the tree 
   @return Whether the tree in the given row and column has burnt out
   */
bool IsBurntOut(const int row, const int col) {
  return Trees[TREE_MAP(row, col, NColsPlusBounds)] >= NMaxBurnSteps;
}

/* Return whether a given tree is on fire

   @param row The row index of the tree 
   @param col The column index of the tree 
   @return Whether the tree in the given row and column is on fire
   */
bool IsOnFire(const int row, const int col) {
  return Trees[TREE_MAP(row, col, NColsPlusBounds)] > 0 &&
    !IsBurntOut(row, col);
}

