# Define the shell to use when executing the make commands
SHELL=/bin/sh # It is recommended to always use /bin/sh

# Define a macro for the name of the executable file
EXECUTABLE=disease.exe

# Define a macro for the compiler command
CC=gcc # Use the GNU compiler

# Define a macro for the compiler flags
CFLAGS=-Wall # Show all warnings
CFLAGS+=-std=gnu99 # Use the C99 standard
CFLAGS+=--pedantic # Be strict about code style

# Define a macro for the source files
SRCS=$(wildcard *.c) # Anything that ends in .c

# Define a macro for the object files
OBJS=$(SRCS:.c=.o) # The corresponding .o files for each of the C source files

# Define a macro for the libraries to link to
LIBS= # No libraries needed

# Define the rule for building the executable
$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Define the rules for compiling the object files
main.o: main.c init.h utils.h globals.h macros.h simulate.h move.h change-states.h print.h update.h
	$(CC) $(CFLAGS) -c -o $@ $<

init.o: init.c init.h utils.h globals.h
	$(CC) $(CFLAGS) -c -o $@ $<

simulate.o: simulate.c simulate.h move.h globals.h macros.h change-states.h print.h update.h
	$(CC) $(CFLAGS) -c -o $@ $<

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c -o $@ $<

move.o: move.c move.h globals.h macros.h
	$(CC) $(CFLAGS) -c -o $@ $<

change-states.o: change-states.c change-states.h globals.h macros.h
	$(CC) $(CFLAGS) -c -o $@ $<

print.o: print.c print.h globals.h macros.h
	$(CC) $(CFLAGS) -c -o $@ $<

update.o: update.c update.h globals.h macros.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Define a rule for removing the object files
clean:
	rm -f $(OBJS)

# Define a rule for removing the object and executable files
distclean:
	rm -f $(OBJS) $(EXECUTABLE)

