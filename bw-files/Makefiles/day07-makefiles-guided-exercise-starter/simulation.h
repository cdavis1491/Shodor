#include <string.h>
#include "utils.h"

/* Declare global parameters */
extern int NRows;
extern int NCols;
extern int BurnProb;
extern bool IsRandFirstTree;

/* Declare other needed global variables */
extern int NRowsPlusBounds; /* Number of rows of trees plus the boundary rows */
extern int MiddleRow; /* The tree in the middle is here. If an even number of
                         rows, this tree is just below the middle */
extern int MiddleCol; /* The tree in the middle is here. If an even number of
                         cols, this tree is just to the right of the middle */
extern int NBurnedTrees; /* The total number of burned trees */
extern int *Trees; /* 1D tree array, contains a boundary around the outside of
                      the forest so the same neighbor checking algorithm can be
                      used on all cells */
extern int *NewTrees; /* Copy of 1D tree array - used so that we don't update
                         the forest too soon as we are deciding which new trees
                         should burn -- does not contain boundary */

/* DECLARE FUNCTIONS */

/* Light a random tree on fire, set all other trees to be not burning */
void InitData();

/* For trees already burning, increment the number of time steps they have
   burned 
   */
void ContinueBurning();

/* Find trees that are not on fire yet and try to catch them on fire from
   burning neighbor trees
   */
void BurnNew();

/* Copy new tree data into old tree data */
void AdvanceTime();

