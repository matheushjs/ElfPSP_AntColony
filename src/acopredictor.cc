#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

/** \file acopredictor.cc */

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::unique_ptr;

ACOPredictor::ACOPredictor(const Config &config)
: dHPChain(config.hp_chain()),
  dCycles(config.cycles()),
  dNAnts(config.n_ants()),
  dAlpha(config.aco_alpha()),
  dBeta(config.aco_beta()),
  dEvap(config.aco_evaporation()),
  dLSFreq(config.ls_frequency()),
  dExchangedAnts(config.exchanged_ants()),
  dRandSeed(config.random_seed()),
  dNMovElems(dHPChain.length() - 2),
  dHCount(0),
  dRandGen(10),
  dRandDist(0.0, 1.0),
  dBestContacts(-1)
{
	dPheromone = new double[dNMovElems*5];
	std::fill(dPheromone, dPheromone + dNMovElems*5, 0.1);

	std::random_device rd;
	for(unsigned i = 0; i < dRandGen.size(); i++){
		if(dRandSeed < 0){
			dRandGen[i].seed(rd() + i);
		} else {
			dRandGen[i].seed(dRandSeed + i);
		}
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
vector<double> ACOPredictor::get_probabilities(int movIndex, vector<double> heuristics) const {
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

	// If sum is 0, would give us division by 0
	if(sum == 0)
		return {0.2, 0.2, 0.2, 0.2, 0.2};

	// sum should not be inf or nan. The user must control this.
	if(std::isinf(sum) || std::isnan(sum)){
		cerr << "ERROR: Encountered unexpected 'Not a Number' or 'Inf'.\n";
		cerr << "Please control the ACO_ALPHA and ACO_BETA parameters more suitably.\n";
		cerr << "Keep in mind that the base for ACO_BETA may be higher than 0, "
		        "and the base for ACO_ALPHA may be very near 0.\n";
		exit(EXIT_FAILURE);
	}

	for(int d = 0; d < 5; d++){
		retval[d] /= sum;
	}

	return retval;
}
