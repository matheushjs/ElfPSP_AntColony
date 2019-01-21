#pragma once

/** \file acopredictor.h */

#include <vector>
#include <random>

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
	std::mt19937 dRandGen;
	std::uniform_real_distribution<> dRandDist;

	double pheromone(int i, int d) const;
	double random();
	std::vector<double> get_heuristics(
			const std::vector<vec3<int> > &possiblePos,
			const std::vector<vec3<int> > &beadVector
		);
	std::vector<double> get_probabilities(int movIndex, std::vector<double> heuristics) const;
	ACOSolution ant_develop_solution();

public:
	/** Default constructor.
	 * \param chain HP chain of the protein whose structure is to be predicted.
	 * \param cycles Number of cycles that should be performed by the ant colony.
	 * \param nAnts Number of ants in the colony. All ants work in each cycle of the colony.
	 * \param alpha Parameter alpha of the pheremone probability equation.
	 * \param beta Parameter beta of the pheremone probability equation.
	 * \param randSeed random seed to pass to the random number generator. If negative, a random seed is chosen.
	 */
	ACOPredictor(const HPChain &chain, int cycles, int nAnts, double alpha, double beta, int randSeed = -1);

	/** The destructor frees memory allocated for holding internal data structures. */
	~ACOPredictor();

	/** Runs the ACO optimization to predict the protein structure.
	 * \return the best solution found by the optimization algorithm. */
	ACOSolution predict();
};
