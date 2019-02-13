#include <iostream>
#include <vector>
#include <limits.h>
#include <stdio.h>

#include "cuda_device_utilities.cuh"
#include "acopredictor_ant_cycle_cuda.cuh"
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
double &CUDAThread::pheromone(int i, int d){
	return pheromones[i*5 + d];
}

__device__
int3 CUDAThread::DIRECTION_VECTOR(int3 prevDir, char dir){
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
int CUDAThread::calculate_contacts(int3 *solution){
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
void CUDAThread::develop_solution(int3 *solution, char *directions){
	for(int i = 0; i < nMovElems; i++){
		int3 prevDir = solution[i+1] - solution[i];
		int3 prevBead = solution[i+1];

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
		// Get number within 0-1
		double unifRand = randomize_d(randNumber);
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
		solution[i+2] = possiblePos[direction];
		directions[i] = direction;
	}
}

__device__
void CUDAThread::solution_from_directions(int3 *solution, char *directions){
	solution[0] = {0,0,0};
	solution[1] = {1,0,0};

	for(int i = 0; i < nMovElems; i++){
		int3 prevDirection = solution[i+1] - solution[i];
		int3 backBead = solution[i+1];
		solution[i+2] = backBead + DIRECTION_VECTOR(prevDirection, directions[i]);
	}
}

__device__
void CUDAThread::local_search(int &solContact, int lsFreq){
	// Copy solution
	for(int i = 0; i < nCoords; i++)
		myOtherDirections[i] = myDirections[i];

	/* DEBUG LOCAL SEARCH
	if(this->tid == 5){
		printf("Solution: ");
		for(int i = 0; i < nCoords; i++){
			print(mySolution[i]);
			printf(" ");
		}
		printf("\n");
	}
	*/

	for(int i = 0; i < lsFreq; i++){
		int idx = randomize_d(randNumber) * nCoords;

		/* DEBUG LOCAL SEARCH
		if(this->tid == 5){
			printf("idx: %d    ", idx);
		}
		*/

		char direction = randomize_d(randNumber) * 5;
		myOtherDirections[idx] = direction;

		/* DEBUG LOCAL SEARCH
		if(this->tid == 5){
			printf("olddir: %d    dir: %d\n", myDirections[idx], direction);
		}
		*/

		solution_from_directions(myOtherSolution, myOtherDirections);
		int contacts = calculate_contacts(myOtherSolution);

		/* DEBUG LOCAL SEARCH
		if(this->tid == 5){
			printf("Generated: ");
			for(int j = 0; j < nCoords; j++){
				print(myOtherSolution[j]);
				printf(" ");
			}
			printf("\n");
		}
		*/

		// Check if is better
		if(contacts > solContact){
			// Update contacts
			solContact = contacts;

			// Update directions
			myDirections[idx] = myOtherDirections[idx];

			// Update solution
			for(int j = 0; j < nCoords; j++)
				mySolution[j] = myOtherSolution[j];
		}
	}
}

__global__
void HostToDevice::ant_develop_solution(
	double *pheromone,
	int nMovElems,
	int3 *solutions,
	int3 *moreSolutions,
	int nCoords,
	char *relDirections,
	char *moreRelDirections,
	int *contacts,
	char *hpChain,
	int lsFreq
){
	CUDAThread *self = new CUDAThread();

	self->tid = blockIdx.x * blockDim.x + threadIdx.x;
	self->randNumber = (13235632^(threadIdx.x*threadIdx.x+77))>>(threadIdx.x%13);

	// Get pointer to our data
	self->mySolution        = solutions + nCoords*self->tid;
	self->myOtherSolution   = moreSolutions + nCoords*self->tid;
	self->myDirections      = relDirections + nMovElems*self->tid;
	self->myOtherDirections = moreRelDirections + nMovElems*self->tid;
	self->pheromones = pheromone;
	self->hpChain    = hpChain;
	self->nCoords    = nCoords;
	self->nMovElems  = nMovElems;

	self->develop_solution(self->mySolution, self->myDirections);

	// Now we calculate contacts
	// If collisions are found, the solution is invalidated (and contacts = -1).
	int nContacts = self->calculate_contacts(self->mySolution);

	// Then we perform local search
	self->local_search(nContacts, lsFreq);

	contacts[self->tid] = nContacts;

	/* DEBUG CONTACTS
	if(self->tid == 0){
		for(int i = 0; i < gridDim.x * blockDim.x; i++){
			printf("%d: %d\n", i, contacts[i]);
		}
	}
	*/

	delete self;

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

__global__
void HostToDevice::find_best_solution(int *contacts, int3 *solutions, int nSolutions, int3 *outSolution, int nCoords){
	// Index of best solutions
	__shared__ int shContacts[1024];
	__shared__ int shIndex[1024];

	int tid = threadIdx.x;
	int stride = blockDim.x;

	/* DEBUG FIND BEST
	if(tid == 0){
		printf("Contacts: ");
		for(int i = 0; i < nSolutions; i++)
			printf("%d ", contacts[i]);
		printf("\n");
	}
	*/

	int maxContacts = -1;
	int maxIndex = -1;

	for(int i = tid; i < nSolutions; i += stride){
		if(contacts[i] > maxContacts){
			maxContacts = contacts[i];
			maxIndex = i;
		}
	}

	shContacts[tid] = maxContacts;
	shIndex[tid] = maxIndex;

	__syncthreads();

	// Reduce within shared memory
	for(int power = 512; power > 0; power /= 2){
		if(tid < power){
			if(shContacts[tid] < shContacts[tid+power]){
				shContacts[tid] = shContacts[tid+power];
				shIndex[tid] = shIndex[tid+power];
			}
		}
		__syncthreads();
	}

	/* DEBUG FIND BEST
	if(tid == 0){
		printf("Best contacts: %d    ", shContacts[0]);
		printf("Best index: %d\n", shIndex[0]);
	}
	*/

	// Copy best solution into out buffer
	int3 *bestSol = solutions + shIndex[0] * nCoords;
	for(int i = tid; i < nCoords; i += stride){
		outSolution[i] = bestSol[i];
	}
}

__global__
void HostToDevice::evaporate_pheromones(double *pheromones, int nMovElems, double evapRate){
	const int nPheromones = nMovElems * 5;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = gridDim.x*blockDim.x;

	for(int i = tid; i < nPheromones; i += stride)
		pheromones[i] *= (1 - evapRate);
}

__global__
void HostToDevice::deposit_pheromones(
	double *pheromones, int nMovElems, char *directions, int *contacts, int hCount
){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	char *myDirections = directions + nMovElems*tid;
	double pheroAmount = contacts[tid] / hCount;

	/* DEBUG DEPOSIT
	if(tid == 5){
		for(int i = 0; i < 5; i++){
			for(int j = 0; j < nMovElems; j++){
				printf("%.3lf ", pheromones[j*5 + i]);
			}
			printf("\n");
		}
	}
	*/

	for(int i = 0; i < nMovElems; i++){
		int d = myDirections[i];
		double *pheroPos = pheromones + (i*5 + d);
		atomicAdd_d(pheroPos, pheroAmount);
	}

	/* DEBUG DEPOSIT
	if(tid == 5){
		for(int i = 0; i < 5; i++){
			for(int j = 0; j < nMovElems; j++){
				printf("%.3lf ", pheromones[j*5 + i]);
			}
			printf("\n");
		}
	}
	*/
}

ACOWithinCUDA::ACOWithinCUDA(
	double *pheromones,
	const string hpChain,
	int nMovElems,
	int nCoords,
	int nAnts,
	int lsFreq,
	int nSols,
	int hCount,
	double evap
) :
	dPheromone(nMovElems*5),
	dSolutions(nCoords*nAnts),
	dMoreSolutions(nCoords*nAnts),
	dRelDirections(nMovElems*nAnts),
	dMoreRelDirections(nMovElems*nAnts),
	dContacts(nAnts),
	dHPChain(hpChain.length()),
	dNMovElems(nMovElems),
	dNCoords(nCoords),
	dNAnts(nAnts),
	dLSFreq(lsFreq),
	dNSolutions(nSols),
	dHCount(hCount),
	dEvap(evap)
{
	// Copy stuff
	dPheromone.memcpyAsync(pheromones);
	dHPChain.memcpyAsync(hpChain.c_str());
	
	int3 fillData[2] = {{0,0,0},{1,0,0}};
	for(int i = 0; i < dNAnts; i++)
		cudaMemcpyAsync(dSolutions.get() + i*nCoords, fillData, sizeof(int3)*2, cudaMemcpyHostToDevice);
}

void ACOWithinCUDA::run(){
	HostToDevice::ant_develop_solution<<<1,dNAnts>>>(dPheromone, dNMovElems,
			dSolutions, dMoreSolutions, dNCoords,
			dRelDirections, dMoreRelDirections,
			dContacts, dHPChain, dLSFreq);

	// We copy best solution into first solution of dMoreSolutions
	HostToDevice::find_best_solution<<<1,1024>>>(dContacts, dSolutions, dNSolutions, dMoreSolutions, dNCoords);

	HostToDevice::evaporate_pheromones<<<1,1024>>>(dPheromone, dNMovElems, dEvap);

	HostToDevice::deposit_pheromones<<<1,dNAnts>>>(dPheromone, dNMovElems, dRelDirections, dContacts, dHCount);
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
	 *   - A second vector of solutions, in which threads can hold "tentative" solutions.
	 * To sinalize error in solutions, we will set the first coordinate to (-1,0,0)
	 */

	int nCoords = dNMovElems + 2;
	string hpChain = dHPChain.get_chain();

	ACOWithinCUDA aco(
			dPheromone, hpChain, dNMovElems, nCoords, dNAnts, dLSFreq,
			dNAnts, dHCount, dEvap);
	aco.run();

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
