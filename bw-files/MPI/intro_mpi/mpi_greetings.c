/**************************************************************
 * Filename: mpi_greetings_2.c
 * Description: This example introduces two more most
 *      important mpi functions, MPI_Send() and MPI_Recv()
 *      All MPI routines starts with MPI_ (MPI_Name_of_routine)
 * How to compile:
 *  $ cc mpi_greetings_2.c -o mpi_greetings_2.exe
 * How to Run on 8 cores:
 *  $ aprun -n 8 ./mpi_greetings_2.exe
 **************************************************************/
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define MASTER 0

int main(int argc, char **argv){
    int my_rank;            //ps rank/id
    int num_ps;             //ps team size
    int distination = MASTER;        // msg distination
    int source;             // msg sourse
    int msg_size;           // msg size in array elements (char)
    int tag = 777;          // flag the message
    char message[200];      // 
    MPI_Status status;      // used for receive function

    MPI_Init(&argc, &argv);  // Start MPI Environment now
    MPI_Comm_size(MPI_COMM_WORLD, &num_ps);     // get team size
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);    // get ps id
    
    //prepare the message that needed to be send!
    sprintf(message, "Greetings Human World: \n"
            "\t|-----------------------|\n"
            "\t| From: ps %d         |_ |\n"
            "\t|        To: ps %d       |\n"
            "\t|_______________________|\n", my_rank, distination);
    
    msg_size = strlen(message)+1;   // Get the length of the message.
    
    //This is where the main action is now happening. 
    //MPI send and receive usually have the if conditional
    //basically how is allowed to send messge and who is to receive.
    if (my_rank == 0){  
        // if you dont use for loop it will only print one message
        for(source = 1; source < num_ps; source++){
            MPI_Recv(message, msg_size, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
            printf("%s \n", message);
        } //End: for loop
    }
    else{
        MPI_Send(&message, msg_size, MPI_CHAR, distination, tag, MPI_COMM_WORLD);
    } //END: if-else
    
    MPI_Finalize();

} //END main()

