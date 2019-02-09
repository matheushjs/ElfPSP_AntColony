#include <iostream>
#include <vector>
#include <cmath>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::string;

/** Makes an ant develop a solution, beginning from the start.
 * Returns the developed solution. */
ACOSolution ACOPredictor::ant_develop_solution() {
	ACOSolution sol;

	for(int i = 0; i < dNMovElems; i++){
		vec3<int> prevDir = sol.previous_direction();
		vec3<int> prevBead = sol.vector().back();

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

		// If all heuristics are 0, there is no possible next direction to take.
		double sum = heurs[0] + heurs[1] + heurs[2] + heurs[3] + heurs[4];
		if(sum == 0){
			sol.set_error();
			return sol;
		}

		// Get probabilities based on the ACO probability equation
		vector<double> probs = get_probabilities(i, heurs);

		/*
		cout << "HorP: " << this->dHPChain.get_chain()[sol.vector().size()] << "\n";
		cout << "Heurs: ";
		for(double i: heurs) cout << i << " ";
		cout << "\n";
		cout << "Probs: ";
		for(double i: probs) cout << i << " ";
		cout << "\n\n";
		*/

		// Accumulate the probability vector
		for(unsigned i = 1; i < probs.size(); i++)
			probs[i] += probs[i-1];

		// Decide direction
		char direction = 0;
		double randomNum = this->random();
		for(unsigned i = 0; i < probs.size(); i++){
			if(randomNum < probs[i]){
				direction = i;
				break;
			}
		}

		// Add bead in the decided direction
		sol.vector().push_back(possiblePos[direction]);
		sol.directions().push_back(direction);
	}

	return sol;
}

