#include <iostream>
#include <vector>
#include <omp.h>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::string;

void ACOPredictor::perform_cycle(vector<ACOSolution> &antsSolutions, int *nContacts){
	// Let each ant develop a solution
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		#pragma omp for
		for(int j = 0; j < dNAnts; j++){
			ACOSolution currentSol = ant_develop_solution();

			if(currentSol.has_error() == false){
				#pragma omp critical
				antsSolutions.push_back(currentSol);
			}
		}

		// Calculate contacts
		#pragma omp for
		for(unsigned j = 0; j < antsSolutions.size(); j++)
			nContacts[j] = antsSolutions[j].count_contacts(dHPChain);

		// Perform local search
		#pragma omp for
		for(unsigned j = 0; j < antsSolutions.size(); j++){
			for(int k = 0; k < dLSFreq; k++){
				ACOSolution tentative = antsSolutions[j];
				int lim = this->random() * tentative.directions().size();
				for(int l = 0; l < lim; l++){
					tentative.perturb_one(dRandGen[tid]);
				}
				int contacts = tentative.count_contacts(dHPChain);
				if(contacts > nContacts[j]){
					antsSolutions[j] = tentative;
				}
			}
		}
	}

	// Check best protein
	for(unsigned j = 0; j < antsSolutions.size(); j++){
		if(nContacts[j] > dBestContacts){
			dBestSol = antsSolutions[j];
			dBestContacts = nContacts[j];
		}
	}

	// Evaporate pheromones
	for(int i = 0; i < dNMovElems; i++){
		for(int j = 0; j < 5; j++){
			pheromone(i, j) *= (1 - dEvap);
		}
	}
	
	// Deposit pheromones
	for(unsigned j = 0; j < antsSolutions.size(); j++)
		ant_deposit_pheromone(antsSolutions[j].directions(), nContacts[j]);
}
