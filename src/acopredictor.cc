#include <iostream>
#include <vector>
#include <cmath>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::string;

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
	
	if(horp == 'P')
		return vector<double>(5, 1.0);

	int contacts[5] = { 0, 0, 0, 0, 0 };
	int collisions[5] = { 0, 0, 0, 0, 0 };

	// Get number of contacts per possible next position
	for(int i = 0; i < 5; i++){
		vec3<int> nextPos = possiblePos[i];
		for(unsigned j = 0; j < beadVector.size(); j++){
			vec3<int> bead = beadVector[j];

			if(nextPos == bead)
				collisions[i]++;

			int norm1 = (nextPos - bead).norm1();
			if(norm1 == 1 && hpchain[j] == 'H')
				contacts[i] += 1;
		}
	}

	for(int i = 0; i < 5; i++){
		if(collisions[i] == 0)
			retval[i] = 1.0 + contacts[i];
		else
			retval[i] = 0.0;
	}

	return retval;
}

/** Return a vector V with 5 probabilities.
 * V[d] is the probability of going to direction d. */
inline vector<double> ACOPredictor::get_probabilities(int movIndex, vector<double> heuristics) const {
	using std::pow;

	vector<double> retval(5);
	double sum = 0;

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

	for(int i = 0; i < dNMovElems; i++){
		vec3<int> prevDir = sol.previous_direction();
		vec3<int> prevBead = sol.dVector.back();

		// Get possible positions for the next bead
		vector<vec3<int>> possiblePos = {
			prevBead + ACOSolution::get_direction_vector(prevDir, 0),
			prevBead + ACOSolution::get_direction_vector(prevDir, 1),
			prevBead + ACOSolution::get_direction_vector(prevDir, 2),
			prevBead + ACOSolution::get_direction_vector(prevDir, 3),
			prevBead + ACOSolution::get_direction_vector(prevDir, 4),
		};

		// Calculate heuristics
		vector<double> heurs = get_heuristics(possiblePos, sol.dVector);

		// If all heuristics are 0, there is no possible next direction to take.
		double sum = heurs[0] + heurs[1] + heurs[2] + heurs[3] + heurs[4];
		if(sum == 0){
			// TODO: BACKTRACK
			sol.dError = true;
			return sol;
		}

		// Get probabilities based on the ACO probability equation
		vector<double> probs = get_probabilities(i, heurs);

		/*
		cout << "HorP: " << this->dHPChain.get_chain()[sol.dVector.size()] << "\n";
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
		sol.dVector.push_back(possiblePos[direction]);
		sol.dDirections.push_back(direction);
	}

	return sol;
}

void ACOPredictor::ant_deposit_pheromone(const ACOSolution &sol){
	const vector<vec3<int>> beads = sol.dVector;
	string &hpchain = dHPChain.get_chain();
	int contacts = 0;

	// First we calculate the protein energy. That is, number of H contacts.
	for(unsigned i = 0; i < beads.size(); i++){
		if(hpchain[i] == 'P') continue;

		for(unsigned j = i+1; j < beads.size(); j++){
			if(hpchain[j] == 'P') continue;
			int norm1 = (beads[i] - beads[j]).norm1();
			if(norm1 == 1) contacts++;
		}
	}

	const vector<char> dirs = sol.dDirections;
	
	// Then we add pheromone
	for(unsigned i = 0; i < dirs.size(); i++){
		pheromone(i, dirs[i]) += contacts / dHCount;
	}
}

/** Each pheromone is multiplied by (1-p) where p is the persistence defined by the user. */
void ACOPredictor::evaporate_pheromone(){
	for(int i = 0; i < dNMovElems; i++){
		for(int j = 0; j < 5; j++){
			pheromone(i, j) *= (1 - dEvap);
		}
	}
}

ACOSolution ACOPredictor::predict(){
	ACOSolution bestSol;

	for(int i = 0; i < dCycles; i++){
		vector<ACOSolution> antsSolutions; // Solutions generated by all ants

		// Let each ant develop a solution
		for(int j = 0; j < dNAnts; j++){
			ACOSolution currentSol = ant_develop_solution();
			if(currentSol.dError == false){
				antsSolutions.push_back(currentSol);
			}
		}

		// Let each ant deposit pheromones over the trail they followed
		for(const ACOSolution &sol: antsSolutions){
			ant_deposit_pheromone(sol);
		}
		
		// Evaporate pheromones
		evaporate_pheromone();

		if(antsSolutions.size() > 0)
			bestSol = antsSolutions.back();

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
	}

	return bestSol;
}
