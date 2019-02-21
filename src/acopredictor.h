#pragma once

/** \file acopredictor.h */

#include <vector>
#include <random>

#include "hpchain.h"
#include "acosolution.h"
#include "config.h"

/** Encapsulates the whole process of performing Ant Colony Optimization to find
 *    a protein conformation with low free energy. */
class ACOPredictor {
	/** @{ */
	/** See the user configuration file for documentation. */
	HPChain dHPChain;
	int dCycles;
	int dNAnts;
	double dAlpha;
	double dBeta;
	double dEvap;
	int dLSFreq;
	int dExchangedAnts;
	int dRandSeed;
	/** @} */

	/** Stores the number of moviments performed by an ant; its value is N-2 where N is the number of beads in the protein. */
	int dNMovElems;
	int dHCount; /**< Stores the number of hydrophobic (H) beads in the protein */
	double *dPheromone; /**< Pheromone matrix. */
	std::vector<std::mt19937> dRandGen; /**< Random number generators used throughout the ACO algorithm. */
	std::uniform_real_distribution<> dRandDist; /**< Random distribution that uses `dRandGen` to generate random numbers. */

	ACOSolution dBestSol; /**< Holds the best solution found by this colony. */
	int dBestContacts; /**< Holds the num of contacts in the best solution. */

	double &pheromone(int i, int d) const;
	double random(int tid = 0);
	std::vector<double> get_heuristics(
			const std::vector<vec3<int>> &possiblePos,
			const std::vector<vec3<int>> &beadVector
		);
	std::vector<double> get_probabilities(int movIndex, std::vector<double> heuristics) const;
	ACOSolution ant_develop_solution(int tid = 0);
	void ant_deposit_pheromone(const std::vector<char> &directions, int nContacts);

	/** Performs a cycle of the ant colony.
	 * A cycle consitst of each:
	 * 1. ant developing a solution,
	 * 2. calculating the solution contacts,
	 * 3. performing local search in each solution,
	 * 4. finding the best solution generated,
	 * 5. evaporating pheromones and finally
	 * 6. depositing pheromones.
	 *
	 * This function is the one parallelized for multiple parallel programming models,
	 * in the files acopredictor_ant_cycle[...]. */
	void perform_cycle(std::vector<ACOSolution> &antsSolutions, int *nContacts);

public:
	/** Constructor from configuration file.
	 * \param config The configuration file object from which we can read user configured-parameters.
	 */
	ACOPredictor(const Config &config);

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

/** Deposits pheromones along the trail followed by the given ant.
 *
 * \param directions Vector of directions followed by the ant.
 * \param nContacts Number of H-H contacts in the protein built by the given ant.
 */
inline
void ACOPredictor::ant_deposit_pheromone(const std::vector<char> &directions, int nContacts){
	for(unsigned i = 0; i < directions.size(); i++){
		pheromone(i, directions[i]) += nContacts / dHCount;
	}
}

/** Returns the pheromone at step i and direction d. */
inline double &ACOPredictor::pheromone(int i, int d) const {
	return dPheromone[i*5 + d];
}

/** Returns a random number in [0,1). */
inline double ACOPredictor::random(int tid) {
	double retval = dRandDist(dRandGen[tid]);
	return retval;
}
