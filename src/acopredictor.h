#pragma once

#include "hpchain.h"
#include "movchain.h"
#include "config.h"

class ACOPredictor {
	HPChain dhpchain;
	const Config &dConfig;
	double *dPheromone;

	double pheromone(int i, int d) const;

public:
	ACOPredictor(const HPChain &chain, const Config &config);
	~ACOPredictor();

	MovChain predict();
};
