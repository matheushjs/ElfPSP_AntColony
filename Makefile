
CFLAGS=-Wall -O2 -I src -std=c++11

MPI_LIBS=-Wl,-Bsymbolic-functions -Wl,-z,relro -L/usr/lib/x86_64-linux-gnu -lmpich
MPI_CFLAGS=-I/usr/include/mpich

# We take as a rule that if any API changes, everything should be rebuilt.
# Same goes for the makefile itself
HARD_DEPS=Makefile hpchain.h config.h vec3.h acopredictor.h acosolution.h

# This is a variable used by Makefile itself
VPATH=src/

all:
	make prog_bt prog_nbt prog_bt_mpi prog_nbt_mpi test

prog_bt_mpi: main.o hpchain.o config.o acopredictor.o acopredictor_predict_mpi.o acopredictor_backtracking.o
	g++ $(CFLAGS) $(MPI_CFLAGS) $^ -o $@ $(MPI_LIBS)

prog_nbt_mpi: main.o hpchain.o config.o acopredictor.o acopredictor_predict_mpi.o acopredictor_nobacktracking.o
	g++ $(CFLAGS) $(MPI_CFLAGS) $^ -o $@ $(MPI_LIBS)

prog_bt: main.o hpchain.o config.o acopredictor.o acopredictor_predict_sequential.o acopredictor_backtracking.o
	g++ $(CFLAGS) $^ -o $@

prog_nbt: main.o hpchain.o config.o acopredictor.o acopredictor_predict_sequential.o acopredictor_nobacktracking.o
	g++ $(CFLAGS) $^ -o $@

test: test.o hpchain.o config.o acopredictor.o acopredictor_predict_sequential.o acopredictor_nobacktracking.o
	g++ $(CFLAGS) $^ -o $@

clean:
	rm -vf *.o
	find -name "*~" -type f -exec rm -vf '{}' \;

docs:
	doxygen Doxyfile

main.o: main.cc $(HARD_DEPS)
test.o: test.cc $(HARD_DEPS)
hpchain.o: hpchain.cc $(HARD_DEPS)
config.o: config.cc $(HARD_DEPS)
acopredictor.o: acopredictor.cc $(HARD_DEPS)
acopredictor_predict_sequential.o: acopredictor_predict_sequential.cc $(HARD_DEPS)
acopredictor_backtracking.o: acopredictor_backtracking.cc $(HARD_DEPS)
acopredictor_nobacktracking.o: acopredictor_nobacktracking.cc $(HARD_DEPS)

acopredictor_predict_mpi.o: acopredictor_predict_mpi.cc $(HARD_DEPS)
	g++ -c $(CFLAGS) $(MPI_CFLAGS) $< -o $@ $(MPI_LIBS)

# Implicit rule for building objects
%.o:
	g++ -c $(CFLAGS) -o "$@" "$<"
