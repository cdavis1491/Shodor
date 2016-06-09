#include "simulation.h"

/* DECLARE FUNCTIONS */

/* Light a random tree on fire, set all other trees to be not burning */
void InitData() {
  int row;
  int col;

  /* Set all trees as having burned for 0 time steps */
  for (row = 1; row < NRows + 1; row++) {
    for (col = 1; col < NCols + 1; col++) {
      Trees[         TREE_MAP(row, col, NColsPlusBounds)] =
        NewTrees[NEW_TREE_MAP(row, col, NCols)]           = 0;
    }
  }

  /* Set the boundaries as burnt out */
  for (row = 0; row < NRowsPlusBounds; row++) {
    /* Left */
    Trees[TREE_MAP(row, 0, NColsPlusBounds)] = NMaxBurnSteps;

    /* Top/Bottom */
    if ((row == 0) || (row == NRows + 1)) {
      for (col = 1; col < NCols + 1; col++) {
        Trees[TREE_MAP(row, col, NColsPlusBounds)] = NMaxBurnSteps;
      }
    }

    /* Right */
    Trees[TREE_MAP(row, NCols + 1, NColsPlusBounds)] = NMaxBurnSteps;
  }

  if (IsRandFirstTree) {
    /* Light a random tree on fire */
    row = RandBetween(1, NRows + 1);
    col = RandBetween(1, NCols + 1);
  }
  else {
    /* Light the middle tree on fire */
    row = MiddleRow + 1;
    col = MiddleCol + 1;
  }
  Trees[         TREE_MAP(row, col, NColsPlusBounds)] =
    NewTrees[NEW_TREE_MAP(row, col, NCols)]           = 1;
  NBurnedTrees++;
}

/* For trees already burning, increment the number of time steps they have
   burned 
   */
void ContinueBurning() {
  int row;
  int col;

  for (row = 1; row < NRows + 1; row++) {
    for (col = 1; col < NCols + 1; col++) {
      if (IsOnFire(row, col)) {
        NewTrees[NEW_TREE_MAP(row, col, NCols)] =
          Trees[     TREE_MAP(row, col, NColsPlusBounds)] + 1;
      }
    }
  }
}

/* Find trees that are not on fire yet and try to catch them on fire from
   burning neighbor trees
   */
void BurnNew() {
  int row;
  int col;

  for (row = 1; row < NRows + 1; row++) {
    for (col = 1; col < NCols + 1; col++) {
      if (!IsOnFire(row, col) && !IsBurntOut(row, col)) {
        /* Check neighbors */
        /* Top */
        if ((IsOnFire(row-1, col) ||
              /* Left */
              IsOnFire(row, col-1) ||
              /* Bottom */
              IsOnFire(row+1, col) ||
              /* Right */
              IsOnFire(row, col+1)) &&
            /* Apply random chance */
            (RandBetween(0, 100) < BurnProb)) {
          /* Catch the tree on fire */
          NewTrees[NEW_TREE_MAP(row, col, NCols)] = 1;

          NBurnedTrees++;
        }
      }
    }
  }
}

/* Copy new tree data into old tree data */
void AdvanceTime() {
  int row;
  int col;

  for (row = 1; row < NRows + 1; row++) {
    for (col = 1; col < NCols + 1; col++) {
      Trees[         TREE_MAP(row, col, NColsPlusBounds)] =
        NewTrees[NEW_TREE_MAP(row, col, NCols)];
    }
  }
}

