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

		ACOSolution localBestSol;
		int localBestContacts = -1;
		
		// Calculate best solution in each thread
		#pragma omp for nowait
		for(unsigned j = 0; j < antsSolutions.size(); j++){
			if(nContacts[j] > localBestContacts){
				localBestSol = antsSolutions[j];
				localBestContacts = nContacts[j];
			}
		}

		// Agglomerate global result
		#pragma omp critical
		if(localBestContacts > dBestContacts){
			dBestSol = localBestSol;
			dBestContacts = localBestContacts;
		}

		// Evaporate pheromones
		#pragma omp for nowait
		for(int i = 0; i < dNMovElems; i++){
			for(int j = 0; j < 5; j++){
				pheromone(i, j) *= (1 - dEvap);
			}
		}

		// Deposit pheromones
		for(unsigned j = 0; j < antsSolutions.size(); j++){
			const vector<char> &directions = antsSolutions[j].directions();
			for(unsigned i = 0; i < directions.size(); i++){
				#pragma omp atomic
				pheromone(i, directions[i]) += nContacts[j] / dHCount;
			}
		}
	} // #pragma omp parallel
}
