#include <iostream>

#include "acopredictor.h"

using std::cout;
using std::cerr;

ACOPredictor::ACOPredictor(const HPChain &hpchain, int cycles, int nAnts, double alpha, double beta)
: dhpchain(hpchain),
  dCycles(cycles),
  dNAnts(nAnts),
  dAlpha(alpha),
  dBeta(beta),
  dNMovElems(hpchain.length() - 2)
{
	dPheromone = new double[dNMovElems*5];
	std::fill(dPheromone, dPheromone + dNMovElems*5, 0.1);

	for(int j = 0; j < 5; j++){
		for(int i = 0; i < hpchain.length()-2; i++){
			cout << pheromone(i, j) << " ";
		} cout << "\n";
	}
}

ACOPredictor::~ACOPredictor(){
	delete[] dPheromone;
}

inline double ACOPredictor::pheromone(int i, int d) const {
	return dPheromone[i*5 + d];
}

MovChain ACOPredictor::predict(){
	for(int i = 0; i < dCycles; i++){
		// Create new chain
		// Iterate over all n-2 directions
		// calculate probabilities
		// decide direction
		if(i%100 == 0) cout << "Cycle: " << i << "\n";
	}

	return MovChain();
}
