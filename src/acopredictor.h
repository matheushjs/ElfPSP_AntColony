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
	std::mt19937 dRandGen; /**< Random number generator used throughout the ACO algorithm. */
	std::uniform_real_distribution<> dRandDist; /**< Random distribution that uses `dRandGen` to generate random numbers. */

	ACOSolution dBestSol; /**< Holds the best solution found by this colony. */
	int dBestContacts; /**< Holds the num of contacts in the best solution. */

	double &pheromone(int i, int d) const;
	double random();
	std::vector<double> get_heuristics(
			const std::vector<vec3<int>> &possiblePos,
			const std::vector<vec3<int>> &beadVector
		);
	std::vector<double> get_probabilities(int movIndex, std::vector<double> heuristics) const;
	ACOSolution ant_develop_solution();
	void ant_deposit_pheromone(const std::vector<char> &directions, int nContacts);
	void develop_solutions(std::vector<ACOSolution> &antsSolutions, int *nContacts);
	void store_best_protein(std::vector<ACOSolution> &antsSolutions, int *nContacts);
	void evaporate_pheromone();

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

inline
void ACOPredictor::develop_solutions(std::vector<ACOSolution> &antsSolutions, int *nContacts){
	// Let each ant develop a solution
	for(int j = 0; j < dNAnts; j++){
		ACOSolution currentSol = ant_develop_solution();
		if(currentSol.has_error() == false){
			antsSolutions.push_back(currentSol);
		}
	}

	// Calculate contacts
	for(unsigned j = 0; j < antsSolutions.size(); j++)
		nContacts[j] = antsSolutions[j].count_contacts(dHPChain);

	// Perform local search
	for(unsigned j = 0; j < antsSolutions.size(); j++){
		for(int k = 0; k < dLSFreq; k++){
			ACOSolution tentative = antsSolutions[j];
			int lim = this->random() * tentative.directions().size();
			for(int l = 0; l < lim; l++){
				tentative.perturb_one(dRandGen);
			}
			int contacts = tentative.count_contacts(dHPChain);
			if(contacts > nContacts[j]){
				antsSolutions[j] = tentative;
			}
		}
	}
}

inline
void ACOPredictor::store_best_protein(std::vector<ACOSolution> &antsSolutions, int *nContacts){
	// Check best protein
	for(unsigned j = 0; j < antsSolutions.size(); j++){
		if(nContacts[j] > dBestContacts){
			dBestSol = antsSolutions[j];
			dBestContacts = nContacts[j];
		}
	}
}

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

/** Each pheromone is multiplied by (1-p) where p is the persistence defined by the user. */
inline
void ACOPredictor::evaporate_pheromone(){
	for(int i = 0; i < dNMovElems; i++){
		for(int j = 0; j < 5; j++){
			pheromone(i, j) *= (1 - dEvap);
		}
	}
}

/** Returns the pheromone at step i and direction d. */
inline double &ACOPredictor::pheromone(int i, int d) const {
	return dPheromone[i*5 + d];
}

/** Returns a random number in [0,1). */
inline double ACOPredictor::random() {
	return dRandDist(dRandGen);
}
