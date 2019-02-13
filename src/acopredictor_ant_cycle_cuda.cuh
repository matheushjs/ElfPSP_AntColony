#pragma once

#include <string>

#include "cuda_device_utilities.cuh"

struct CUDAThread {
	int tid;
	int randNumber;
	int3 *mySolution;
	int3 *myOtherSolution;
	char *myDirections;
	char *myOtherDirections;
	double *pheromones;
	char *hpChain;
	int nCoords;
	int nMovElems;

	static __device__ int3 DIRECTION_VECTOR(int3 prevDir, char dir);

	__device__ void solution_from_directions(int3 *solution, char *directions);
	__device__ void local_search(int &solContact, int lsFreq);

	__device__ double &pheromone(int i, int d);
	__device__ int calculate_contacts(int3 *solution);
	__device__ void develop_solution(int3 *solution, char *directions);
};

namespace HostToDevice{
	__global__
	void ant_develop_solution(
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
	);

	__global__
	void find_best_solution(int *contacts, int3 *solutions, int nSolutions, int3 *outSolution, int nCoords);

	__global__
	void evaporate_pheromones(double *pheromones, int nMovElems, double evapRate);

	__global__
	void deposit_pheromones(double *pheromones, int nMovElems, char *directions, int *contacts, int hCount);

}

class ACOWithinCUDA {
	CUDAPointer<double> dPheromone;
	CUDAPointer<int3>   dSolutions;
	CUDAPointer<int3>   dMoreSolutions;
	CUDAPointer<char>   dRelDirections;
	CUDAPointer<char>   dMoreRelDirections;
	CUDAPointer<int>    dContacts;
	CUDAPointer<char>   dHPChain;
	int dNMovElems;
	int dNCoords;
	int dNAnts;
	int dLSFreq;
	int dNSolutions;
	int dHCount;
	double dEvap;


public:
	ACOWithinCUDA(double *pheromones, const std::string hpChain, int nMovElems,
			int nCoords, int nAnts, int lsFreq, int nSols, int hCount, double evap);
	void run();
};
