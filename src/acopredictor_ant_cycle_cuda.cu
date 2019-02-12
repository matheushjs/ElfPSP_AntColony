#include <iostream>
#include <vector>
#include <limits.h>
#include <stdio.h>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::string;

__device__ const char UP    = 0;
//__device__ const char DOWN  = 1;
__device__ const char LEFT  = 2;
__device__ const char RIGHT = 3;
__device__ const char FRONT = 4;

__device__
void print(int3 a){
	printf("(%d, %d, %d)", a.x, a.y, a.z);
}

__device__
int3 *get_solution(int3 *solutions, int idx, int nCoords){
	return solutions + (nCoords * idx);
}

__device__
char *get_rel_directions(char *relDirections, int idx, int nMovElems){
	return relDirections + idx*nMovElems;
}

__device__
double get_pheromone(double *pheromone, int i, int d){
	return pheromone[i*5 + d];
}

__device__
int3 *get_possiblePositions(int3 *possiblePositions, int idx){
	return possiblePositions + idx*5;
}

__device__
int3 operator-(int3 a, int3 b){
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__
int3 operator+(int3 a, int3 b){
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__
int norm1(int3 a){
	return abs(a.x) + abs(a.y) + abs(a.z);
}

__device__
void randomize(int &randNumber){
	randNumber ^= (randNumber << 13);
	randNumber ^= (randNumber >> 17);
	randNumber ^= (randNumber << 15);
}

__device__
double normal_rand(int randNumber){
	return (unsigned int) randNumber / (double) UINT_MAX;
}

__device__
int3 DIRECTION_VECTOR(int3 prevDir, char dir){
	if(dir == FRONT){
		return prevDir;
	}

	// Will be either 1 or -1
	int sign = prevDir.x + prevDir.y + prevDir.z;

	struct { char x, z; }
		isZero = {
			.x = (prevDir.x == 0),
			.z = (prevDir.z == 0),
		};
	
	int3 retval = {0,0,0};
	if(dir == RIGHT){
		retval.x = sign * isZero.x;
		retval.y = sign * !isZero.x;
	} else if(dir == LEFT){
		retval.x = -sign * isZero.x;
		retval.y = -sign * !isZero.x;
	} else if(dir == UP){
		retval.z = 1 * isZero.z;
		retval.y = 1 * !isZero.z;
	} else /* if(dir == DOWN) */ {
		retval.z = -1 * isZero.z;
		retval.y = -1 * !isZero.z;
	}

	return retval;
}

__device__
int3 previous_direction(int3 *solution, int nCoords, int progress){
	int3 back = solution[progress+1];
	int3 backback = solution[progress];
	return back - backback;
}

__device__
int3 solution_back(int3 *solution, int nCoords, int progress){
	return solution[progress+1];
}

__device__
int calculate_contacts(int3 *solution, int nCoords, char *hpChain){
	int nContacts = 0;
	for(int i = 0; i < nCoords; i++){
		if(hpChain[i] == 'P') continue;
		for(int j = i+1; j < nCoords; j++){
			int norm = norm1(solution[i] - solution[j]);
			if(norm == 0){
				// Invalidate solution
				solution[0].x = -1;
				return -1;
			} else if(hpChain[j] == 'H' && norm == 1){
				nContacts++;
			}
		}
	}
	return nContacts;
}

__device__
void develop_solution(int3 *solution, int nCoords, char *myDirections, int nMovElems){
	// TODO: Improve RNG!!! This won't even work across kernel calls.
	int randNumber = (13235632^(threadIdx.x*threadIdx.x+77))>>(threadIdx.x%13);
	int progress = 0;
	
	for(int i = 0; i < nMovElems; i++){
		int3 prevDir = previous_direction(solution, nCoords, progress);
		int3 prevBead = solution_back(solution, nCoords, progress);

		/*
		print(prevDir);
		print(prevBead);
		printf("\n");
		*/

		int3 possiblePos[5] = {
			prevBead + DIRECTION_VECTOR(prevDir, 0),
			prevBead + DIRECTION_VECTOR(prevDir, 1),
			prevBead + DIRECTION_VECTOR(prevDir, 2),
			prevBead + DIRECTION_VECTOR(prevDir, 3),
			prevBead + DIRECTION_VECTOR(prevDir, 4)
		};
		/*
		print(possiblePos[0]);
		print(possiblePos[1]);
		print(possiblePos[2]);
		print(possiblePos[3]);
		print(possiblePos[4]);
		printf("\n");
		*/

		// TODO: get heuristics
		double heurs[5] = {1, 1, 1, 1, 1};

		// If all heuristics are 0, there is no possible next direction to take.
		double sum = heurs[0] + heurs[1] + heurs[2] + heurs[3] + heurs[4];
		if(sum == 0)
			solution[0].x = -1; // Signalizes error
		
		// TODO: get probabilities
		double probs[5] = {0.2, 0.2, 0.2, 0.2, 0.2};

		// Accumulate the probability vector
		for(int j = 1; j < 5; j++)
			probs[j] += probs[j-1];

		// Decide direction
		char direction = 0;
		randomize(randNumber);
		// Get number within 0-1
		double unifRand = normal_rand(randNumber);
		for(int j = 0; j < 5; j++){
			if(unifRand < probs[j]){
				direction = j;
				break;
			}
		}

		/*
		printf("Decided: %d: ", direction);
		print(possiblePos[direction]);
		printf("\n");
		*/

		// Must be offset by 2, cuz the first 2 are (0,0,0) and (1,0,0) and we disconsider them
		solution[progress+2] = possiblePos[direction];
		myDirections[progress] = direction;
		progress++;
	}
}


__global__
void ant_develop_solution_device(
	double *pheromone,
	int nMovElems,
	int3 *solutions,
	int nCoords,
	char *relDirections,
	int3 *possiblePositions,
	int *contacts,
	char *hpChain
){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Get pointer to our data
	int3 *mySolution   = get_solution(solutions, tid, nCoords);
	char *myDirections = get_rel_directions(relDirections, tid, nMovElems);

	develop_solution(mySolution, nCoords, myDirections, nMovElems);

	// Now we calculate contacts
	// If collisions are found, the solution is invalidated (and contacts = -1).
	int nContacts = calculate_contacts(mySolution, nCoords, hpChain);

	// Then we perform local search


	contacts[tid] = nContacts;

	if(tid == 0){
		for(int i = 0; i < gridDim.x * blockDim.x; i++){
			printf("%d: %d\n", i, contacts[i]);
		}
	}


	/* DEBUG PROTEINS PRODUCED
	for(int i = 0; i < gridDim.x * blockDim.x; i++){
		if(tid == i){
			int3 *mySolution = get_solution(solutions, tid, nCoords);
			for(int j = 0; j < nMovElems+2; j++){
				print(mySolution[j]);
				printf(" ");
			}
			printf("\n");
		}
		__syncthreads();
	}
	*/

	/* DEBUG PROGRESS
	for(int i = 0; i < 10; i++)
		printf("%d ", progress[i]);
	printf("\n");
	*/

	/* DEBUG SOLUTION VECTORS
	for(int i = 0; i < 10; i++){
		int3 a = solutions[nCoords*i];
		int3 b = solutions[nCoords*i+1];

		printf("(%d, %d, %d), (%d, %d, %d)\n",
			a.x, a.y, a.z, b.x, b.y, b.z);
	}
	*/

	/* DEBUG PHEROMONES
	for(int j = 0; j < 5; j++){
		for(int i = 0; i < nMovElems; i++){
			printf("%lf ", get_pheromone(pheromone, i, j));
		}
		printf("\n");
	}
	*/
}

void ACOPredictor::perform_cycle(vector<ACOSolution> &antsSolutions, int *nContacts){
	/* Data we need in the GPU:
	 *   - pheromone matrix
	 *   - Solutions initialized with (0,0,0) (1,0,0), and with enough space for N coordinates total
	 *   - Vectors with relative directions adopted for each solution
	 *   - Vector for tracking progress for each solution
	 *   - Vector of 5 possible next positions, for each solution
	 *   - Vector of contact count, for each solution. We sinalize lost proteins with negative contacts.
	 *   - Vector of HP chain
	 * To sinalize error in solutions, we will set the first coordinate to (-1,0,0)
	 */
	int nCoords = dNMovElems + 2;
	string hpChain = dHPChain.get_chain();

	double *d_pheromone;         cudaMalloc(&d_pheromone, sizeof(double)*dNMovElems*5);
	int3   *d_solutions;         cudaMalloc(&d_solutions, sizeof(int3)*nCoords*dNAnts);
	char   *d_relDirections;     cudaMalloc(&d_relDirections, sizeof(char)*dNAnts*dNMovElems);
	int3   *d_possiblePositions; cudaMalloc(&d_possiblePositions, sizeof(int3)*5*dNAnts);
	int    *d_contacts;          cudaMalloc(&d_contacts, sizeof(int)*dNAnts);
	char   *d_hpChain;           cudaMalloc(&d_hpChain, sizeof(char)*hpChain.length());

	// Copy stuff
	cudaMemcpyAsync(d_pheromone, dPheromone, sizeof(double)*dNMovElems*5, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_hpChain, hpChain.c_str(), sizeof(char)*hpChain.length(), cudaMemcpyHostToDevice);

	int3 fillData[2] = {{0,0,0},{1,0,0}};
	for(int i = 0; i < dNAnts; i++)
		cudaMemcpyAsync(d_solutions + i*nCoords, fillData, sizeof(int3)*2, cudaMemcpyHostToDevice);

	// Call device function
	ant_develop_solution_device<<<1,dNAnts>>>(d_pheromone, dNMovElems, d_solutions,
			nCoords, d_relDirections, d_possiblePositions, d_contacts,
			d_hpChain);

	// Free stuff
	cudaFree(d_pheromone);
	cudaFree(d_solutions);
	cudaFree(d_relDirections);
	cudaFree(d_possiblePositions);
	cudaFree(d_contacts);
	cudaFree(d_hpChain);

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
				tentative.perturb_one(dRandGen[0]);
			}
			int contacts = tentative.count_contacts(dHPChain);
			if(contacts > nContacts[j]){
				antsSolutions[j] = tentative;
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
