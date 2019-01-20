#include <iostream>
#include <vector>
#include <cmath>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;

ACOPredictor::ACOPredictor(const HPChain &hpchain, int cycles, int nAnts, double alpha, double beta, int randSeed)
: dhpchain(hpchain),
  dCycles(cycles),
  dNAnts(nAnts),
  dAlpha(alpha),
  dBeta(beta),
  dNMovElems(hpchain.length() - 2),
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

	for(int i = 0; i < 10; i++){
		cout << this->random() << "\n";
	}
}

ACOPredictor::~ACOPredictor(){
	delete[] dPheromone;
}

/** Returns the pheromone at step i and direction d. */
inline double ACOPredictor::pheromone(int i, int d) const {
	return dPheromone[i*5 + d];
}

/** Returns a random number in [0,1). */
inline double ACOPredictor::random() {
	return dRandDist(dRandGen);
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
inline ACOSolution ACOPredictor::ant_develop_solution() const {
	ACOSolution sol;

	vector<double> heurs(5, 1.0);

	for(int i = 0; i < dNMovElems; i++){
		// Calculate heuristics

		vector<double> probs = get_probabilities(i, heurs);

		// decide direction
	}

	return sol;
}

ACOSolution ACOPredictor::predict(){
	ACOSolution bestSol();

	for(int i = 0; i < dCycles; i++){
		vector<ACOSolution> antsSolutions; // Solutions generated by all ants

		for(int j = 0; j < dNAnts; j++){
			ACOSolution currentSol = ant_develop_solution();
			antsSolutions.push_back(currentSol);
		}

		if(i%100 == 0) cout << "Cycle: " << i << "\n";
	}

	return ACOSolution();
}
