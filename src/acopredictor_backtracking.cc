#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <stack>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::stack;

ACOPredictor::ACOPredictor(
	const HPChain &hpchain, int cycles, int nAnts,
	double alpha, double beta, double evap, int randSeed)
: dHPChain(hpchain),
  dCycles(cycles),
  dNAnts(nAnts),
  dAlpha(alpha),
  dBeta(beta),
  dEvap(evap),
  dNMovElems(hpchain.length() - 2),
  dHCount(0),
  dRandDist(0.0, 1.0)
{
	dPheromone = new double[dNMovElems*5];
	std::fill(dPheromone, dPheromone + dNMovElems*5, 0.1);

	if(randSeed < 0){
		std::random_device rd;
		dRandGen.seed(rd());
	} else {
		dRandGen.seed(randSeed);
	}

	// Count number of H residues
	for(char c: dHPChain.get_chain()){
		if(c == 'H')
			dHCount++;
	}

	if(dHCount == 0){
		cerr << "Number of H beads cannot be 0!!!\n";
		std::exit(0);
	}
}

ACOPredictor::~ACOPredictor(){
	delete[] dPheromone;
}

/** Returns the pheromone at step i and direction d. */
inline double &ACOPredictor::pheromone(int i, int d) const {
	return dPheromone[i*5 + d];
}

/** Returns a random number in [0,1). */
inline double ACOPredictor::random() {
	return dRandDist(dRandGen);
}

/** Returns a vector V with 5 heuristics.
 * V[d] is the heuristic associated with direction d. */
inline vector<double>
ACOPredictor::get_heuristics(const vector<vec3<int>> &possiblePos, const vector<vec3<int>> &beadVector){
	string hpchain = this->dHPChain.get_chain();
	vector<double> retval(5, 0);

	// Bead being added is H or P?
	char horp = hpchain[beadVector.size()];
	
	int contacts[5] = { 0, 0, 0, 0, 0 };
	int collisions[5] = { 0, 0, 0, 0, 0 };

	// Get number of contacts per possible next position
	// Here we assume bead is hydrophobic
	for(int i = 0; i < 5; i++){
		vec3<int> nextPos = possiblePos[i];
		for(unsigned j = 0; j < beadVector.size(); j++){
			int norm1 = (nextPos - beadVector[j]).norm1();

			if(norm1 == 0){
				collisions[i]++;
			} else if(norm1 == 1 && hpchain[j] == 'H'){
				contacts[i] += 1;
			}
		}
	}

	// If bead is P, we disregard the 'contacts' vector
	if(horp == 'P'){
		for(int i = 0; i < 5; i++){
			if(collisions[i] == 0)
				retval[i] = 1.0;
			else
				retval[i] = 0.0;
		}
	} else {
		for(int i = 0; i < 5; i++){
			if(collisions[i] == 0)
				retval[i] = 1.0 + contacts[i];
			else
				retval[i] = 0.0;
		}
	}

	return retval;
}

/** Return a vector V with 5 probabilities.
 * V[d] is the probability of going to direction d. */
inline vector<double> ACOPredictor::get_probabilities(int movIndex, vector<double> heuristics) const {
	using std::pow;

	// If all heuristics are 0, it would give us a division by 0.
	double sum = heuristics[0] + heuristics[1] + heuristics[2] + heuristics[3] + heuristics[4];
	if(sum == 0)
		return {0, 0, 0, 0, 0};

	vector<double> retval(5);
	sum = 0;

	for(int d = 0; d < 5; d++){
		double A = pow(pheromone(movIndex, d), dAlpha);
		double B = pow(heuristics[d], dBeta);
		double aux = A * B;

		sum += aux;
		retval[d] = aux;
	}

	for(int d = 0; d < 5; d++){
		retval[d] /= sum;
	}

	return retval;
}

/** Makes an ant develop a solution, beginning from the start.
 * Returns the developed solution. */
ACOSolution ACOPredictor::ant_develop_solution() {
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
		double rand = this->random();
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

/** Deposits pheromones along the trail followed by the given ant.
 *
 * \param directions Vector of directions followed by the ant.
 * \param nContacts Number of H-H contacts in the protein built by the given ant.
 */
inline
void ACOPredictor::ant_deposit_pheromone(const vector<char> &directions, int nContacts){
	for(unsigned i = 0; i < directions.size(); i++){
		pheromone(i, directions[i]) += nContacts / dHCount;
	}
}

/** Each pheromone is multiplied by (1-p) where p is the persistence defined by the user. */
inline
void ACOPredictor::evaporate_pheromone(){
	for(int i = 0; i < dNMovElems; i++){
		for(int j = 0; j < 5; j++){
			pheromone(i, j) *= (1 - dEvap);
		}
	}
}

struct ACOPredictor::Results ACOPredictor::predict(){
	ACOSolution bestSol;
	int bestContacts = -1;

	for(int i = 0; i < dCycles; i++){
		vector<ACOSolution> antsSolutions; // Solutions generated by all ants

		// Let each ant develop a solution
		for(int j = 0; j < dNAnts; j++){
			ACOSolution currentSol = ant_develop_solution();
			if(currentSol.has_error() == false){
				antsSolutions.push_back(currentSol);
			}
		}

		// Calculate contacts
		std::unique_ptr<int[]> nContacts(new int[antsSolutions.size()]);
		for(unsigned j = 0; j < antsSolutions.size(); j++)
			nContacts[j] = antsSolutions[j].count_contacts(dHPChain);

		// Check best protein
		for(unsigned j = 0; j < antsSolutions.size(); j++){
			if(nContacts[j] > bestContacts){
				bestSol = antsSolutions[j];
				bestContacts = nContacts[j];
			}
		}

		// Deposit pheromones
		for(unsigned j = 0; j < antsSolutions.size(); j++)
			ant_deposit_pheromone(antsSolutions[j].directions(), nContacts[j]);

		// Evaporate pheromones
		evaporate_pheromone();

		/*
		if(i%2 == 0){
			for(int j = 0; j < 5; j++){
				for(int i = 0; i < dNMovElems; i++)
					cout << pheromone(i, j) << " ";
				cout << "\n";
			}
			cout << "\n";
		}
		*/

		// if(i%5 == 0) cout << "Cycle: " << i << "\n";
	}

	Results res = {
		.solution = bestSol,
		.contacts = bestContacts
	};

	return res;
}
