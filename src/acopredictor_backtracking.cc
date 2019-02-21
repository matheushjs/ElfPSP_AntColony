#include <iostream>
#include <vector>
#include <cmath>
#include <stack>

#include "acopredictor.h"

/** \file acopredictor_backtracking.cc */

using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::stack;

/** Makes an ant develop a solution, beginning from the start.
 * Returns the developed solution. */
ACOSolution ACOPredictor::ant_develop_solution(int tid) {
	ACOSolution sol;

	struct State {
		vector<double> probs; /* Vector of 5 probabilities. */
		vector<vec3<int>> possPos; /* Vector of 5 possible "next" positions. */
	};

	/* We will use a stack for backtracking.
	 * The stack holds probabilities of following each direction. */
	stack<State> stk;

	/* The first direction is completely random. */
	vec3<int> prevDir = sol.previous_direction();
	vec3<int> prevBead = sol.vector().back();
	State initial;
	initial.probs = {0.2, 0.2, 0.2, 0.2, 0.2};
	initial.possPos = {
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 0),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 1),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 2),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 3),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 4)
	};
	stk.push(initial);

	while(true){
		/* If stack is empty, protein can't be folded.
		 * This can't happen, so there is a bug in the program. */
		if(stk.size() == 0){
			cerr << "Stack reached size 0. This must have been caused by a bug.\n";
			std::exit(1);
		}

		/* If stack reached size dNMovElems, then we fully formed a protein. */
		if(stk.size() == (unsigned) dNMovElems){
			break;
		}

		State &current = stk.top(); /* This has to be a reference! */
		
		/* Calculate sum as we will need it many times. */
		double sum = 0.0;
		for(double d: current.probs)
			sum += d;

		/* If all possibilities of current state are 0, we have to pop stack. */
		if(sum == 0){
			stk.pop();
			sol.vector().pop_back();
			sol.directions().pop_back();
			continue;
		}

		/* Normalize probabilities, as they might be unnormalized */
		for(unsigned i = 0; i < 5; i++)
			current.probs[i] /= sum;

		/* Accumulate vector of probabilities. We will need for choosing next direction. */
		double accumulated[5] = { current.probs[0] }; // Initialize just first member
		for(unsigned i = 1; i < 5; i++)
			accumulated[i] = accumulated[i-1] + current.probs[i];

		/* Decide next direction. */
		char dir = -1;
		double rand = this->random(tid);
		for(unsigned i = 0; i < 5; i++){
			if(rand <= accumulated[i]){
				dir = i;
				break;
			}
		}

		/* Add bead in the decided direction. */
		sol.directions().push_back(dir);
		sol.vector().push_back(current.possPos[dir]);

		/* Mark current direction as already taken. */
		current.probs[dir] = 0.0;

		/* Calculate state to push into the stack. */
		vec3<int> prevDir = sol.previous_direction();
		vec3<int> prevBead = sol.vector().back();
		State next;

		// Get possible positions for the next bead
		vector<vec3<int>> possiblePos = {
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 0),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 1),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 2),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 3),
			prevBead + ACOSolution::DIRECTION_VECTOR(prevDir, 4),
		};

		// Calculate heuristics
		vector<double> heurs = get_heuristics(possiblePos, sol.vector());

		// Get probabilities based on the ACO probability equation
		vector<double> probs = get_probabilities(stk.size()-1, heurs);

		// Push state in the stack
		next.probs = probs;
		next.possPos = possiblePos;
		stk.push(next);
	}

	return sol;
}

