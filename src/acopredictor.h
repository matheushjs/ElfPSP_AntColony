#pragma once

#include "hpchain.h"
#include "movchain.h"

class ACOPredictor {
	HPChain dhpchain;
	double *dPheromone;

	double pheromone(int i, int d) const;

public:
	ACOPredictor(const HPChain &chain);
	~ACOPredictor();

	MovChain predict();
};
