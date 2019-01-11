#pragma once

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

public:
	ACOPredictor(const HPChain &chain, int cycles, int nAnts, double alpha, double beta);
	~ACOPredictor();

	ACOSolution predict();
};
