#include <stdlib.h>

/* Declare other needed global variables */                                     
extern int NTrees; /* Total number of trees in the forest */
extern int NTreesPlusBounds; /* Total number of trees plus the boundaries */
extern int *Trees; /* 1D tree array, contains a boundary around the outside of
                      the forest so the same neighbor checking algorithm can be 
                      used on all cells */
extern int *NewTrees; /* Copy of 1D tree array - used so that we don't update
                         the forest too soon as we are deciding which new trees
                         should burn -- does not contain boundary */

/* DECLARE FUNCTIONS */

/* Allocate dynamic memory */
void AllocateMemory();

/* Free allocated memory */
void FreeMemory();

