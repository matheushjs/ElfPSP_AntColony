#pragma once

/** \file acopredictor.h */

#include <vector>
#include <random>

#include "hpchain.h"
#include "acosolution.h"

/** Encapsulates the whole process of performing Ant Colony Optimization to find
 *    a protein conformation with low free energy. */
class ACOPredictor {
	/** @{ */
	/** See constructor ACOPredictor(). */
	HPChain dHPChain;
	int dCycles;
	int dNAnts;
	double dAlpha;
	double dBeta;
	double dEvap;
	/** @} */

	/** Stores the number of moviments performed by an ant; its value is N-2 where N is the number of beads in the protein. */
	int dNMovElems;
	int dHCount; /**< Stores the number of hydrophobic (H) beads in the protein */
	double *dPheromone; /**< Pheromone matrix. */
	std::mt19937 dRandGen; /**< Random number generator used throughout the ACO algorithm. */
	std::uniform_real_distribution<> dRandDist; /**< Random distribution that uses `dRandGen` to generate random numbers. */

	double &pheromone(int i, int d) const;
	double random();
	std::vector<double> get_heuristics(
			const std::vector<vec3<int>> &possiblePos,
			const std::vector<vec3<int>> &beadVector
		);
	std::vector<double> get_probabilities(int movIndex, std::vector<double> heuristics) const;
	ACOSolution ant_develop_solution();
	void ant_deposit_pheromone(const std::vector<char> &directions, int nContacts);
	void evaporate_pheromone();

public:
	/** Default constructor.
	 * \param chain HP chain of the protein whose structure is to be predicted.
	 * \param cycles Number of cycles that should be performed by the ant colony.
	 * \param nAnts Number of ants in the colony. All ants work in each cycle of the colony.
	 * \param alpha Parameter alpha of the pheremone probability equation.
	 * \param beta Parameter beta of the pheremone probability equation.
	 * \param evap Evaporation rate of pheromones. Each pheromone is multiplied by (1-evap) at each iteration.
	 * \param randSeed random seed to pass to the random number generator. If negative, a random seed is chosen.
	 */
	ACOPredictor(const HPChain &chain, int cycles, int nAnts,
	             double alpha, double beta, double evap, int randSeed = -1);

	/** The destructor frees memory allocated for holding internal data structures. */
	~ACOPredictor();

	/** Structures used to return results to the callee. */
	struct Results {
		ACOSolution solution; /**< Best protein obtained by the ACO algorithm. */
		int contacts; /**< Number of H-H contacts within `solution`. */
	};

	/** Runs the ACO optimization to predict the protein structure.
	 * \return the best solution found by the optimization algorithm. */
	struct Results predict();
};
