#pragma once

#include <vector>

#include "hpchain.h"
#include "acosolution.h"

class ACOPredictor {
	HPChain dhpchain;
	int dCycles;
	int dNAnts;
	double dAlpha;
	double dBeta;
	int dNMovElems;
	double *dPheromone;

	double pheromone(int i, int d) const;
	std::vector<double> get_probabilities(int movIndex, std::vector<double> heuristics) const;
	ACOSolution ant_develop_solution() const;

public:
	ACOPredictor(const HPChain &chain, int cycles, int nAnts, double alpha, double beta);
	~ACOPredictor();

	ACOSolution predict();
};
