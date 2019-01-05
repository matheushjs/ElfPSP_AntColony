
CFLAGS=-Wall -O3 -I src -std=c++11

# We take as a rule that if any API changes, everything should be rebuilt.
# Same goes for the makefile itself
HARD_DEPS=Makefile hpchain.h config.h vec3.h movchain.h acopredictor.h

# This is a variable used by Makefile itself
VPATH=src/

all:
	make prog test

prog: main.o hpchain.o config.o movchain.o acopredictor.o
	g++ $(CFLAGS) $^ -o $@

test: test.o hpchain.o config.o movchain.o acopredictor.o
	g++ $(CFLAGS) $^ -o $@

clean:
	rm -vf *.o
	find -name "*~" -type f -exec rm -vf '{}' \;

main.o: main.cc $(HARD_DEPS)
test.o: test.cc $(HARD_DEPS)
hpchain.o: hpchain.cc $(HARD_DEPS)
movchain.o: movchain.cc $(HARD_DEPS)
config.o: config.cc $(HARD_DEPS)
acopredictor.o: acopredictor.cc $(HARD_DEPS)

# Implicit rule for building objects
%.o:
	g++ -c $(CFLAGS) -o "$@" "$<"
