/**************************************************************
 * Filename: arrays.c
 * Autor: Mobeen Ludin
 * Disreption: first c program. This simple program basically
 *      prints a greetings message to screen"
 *
 * How Compile: gcc -g arrays.c -o arrays.exe
 * How to Run: ./arrays.exe
 *************************************************************/
# include <stdlib.h>
# include <stdio.h>

int main ( ){
  float student[35]; 

  double a[6][2] = { {20.00, 15.00}, {6.5, 5.27}, {4.69, 4.84},
                        {3.08, 6.00}, {0.99, 10.83}, {0.02, 27.43} };
  
  double b[2][3] = { {20.0, 0.25, 0.30}, {18.0, 0.00, 0.20} };
  
  double c[6][3];
  int i, j, k;
  
  printf ( "Compute matrix product C = A * B.\n" );

  for ( i = 0; i < 6; i++ )
  {
    for ( j = 0; j < 3; j++ )
    {
      c[i][j] = 0.0;
      for ( k = 0; k < 2; k++ )
      {
        c[i][j] = c[i][j] + a[i][k] * b[k][j];
      }
    }
  }

  printf( "\n|--------------A------------|" );
  for (i = 0; i < 6; i++){
      printf(" \n");
      for (j = 0; j< 2; j++){
        printf ( "  %2.3f  ",a[i][j]);
        }
  }
  printf( "\n|-------------*-------------|" );
  printf( "\n|-------------B-------------|" );
  
  for (i = 0; i < 2; i++){
      printf(" \n");
      for (j = 0; j< 3; j++){
        printf (" %2.3f  ",b[i][j]);
        }
  }
  
  printf( "\n|-------------=-------------|" );
  for (i = 0; i < 6; i++){
      printf(" \n");
      for (j = 0; j< 3; j++){
        printf ( " %2.3f  ",c[i][j]);
        }
  }
  printf( "\n|-------------C------------|\n" );
  
  return 0;
}
