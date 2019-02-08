#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::unique_ptr;

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

/** Returns a vector V with 5 heuristics.
 * V[d] is the heuristic associated with direction d. */
vector<double>
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
			vec3<int> bead = beadVector[j];

			if(nextPos == bead)
				collisions[i]++;

			int norm1 = (nextPos - bead).norm1();
			if(norm1 == 1 && hpchain[j] == 'H')
				contacts[i] += 1;
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
vector<double> ACOPredictor::get_probabilities(int movIndex, vector<double> heuristics) const {
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
