#pragma once

/** \file acopredictor.h */

#include <vector>

#include "hpchain.h"
#include "acosolution.h"

/** Encapsulates the whole process of performing Ant Colony Optimization to find
 *    a protein conformation with low free energy. */
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
	/** Default constructor.
	 * \param chain HP chain of the protein whose structure is to be predicted.
	 * \param cycles Number of cycles that should be performed by the ant colony.
	 * \param nAnts Number of ants in the colony. All ants work in each cycle of the colony.
	 * \param alpha Parameter alpha of the pheremone probability equation.
	 * \param beta Parameter beta of the pheremone probability equation.
	 */
	ACOPredictor(const HPChain &chain, int cycles, int nAnts, double alpha, double beta);

	/** The destructor frees memory allocated for holding internal data structures. */
	~ACOPredictor();

	/** Runs the ACO optimization to predict the protein structure.
	 * \return the best solution found by the optimization algorithm. */
	ACOSolution predict();
};
