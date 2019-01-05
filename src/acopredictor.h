#pragma once

#include "hpchain.h"
#include "movchain.h"

class ACOPredictor {
	HPChain hpchain;

public:
	ACOPredictor(const HPChain &chain);

	MovChain predict();
};
