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
		double *pheromone,   int nMovElems,
		int3 *solutions,     int3 *moreSolutions, int nCoords,
		char *relDirections, char *moreRelDirections,
		int *contacts,       char *hpChain,       int lsFreq
	);

	__global__
	void find_best_solution(
		int *contacts,       int nContacts,
		char *directions,    int nMovElems,
		char *outDirections, int *outBestContact
	);

	__global__
	void evaporate_pheromones(double *pheromones, int nMovElems, double evapRate);

	__global__
	void deposit_pheromones(double *pheromones, int nMovElems, char *directions, int *contacts, int hCount);
}

struct ACODeviceData {
	CUDAPointer<double> pheromone;
	CUDAPointer<int3>   solutions;
	CUDAPointer<int3>   moreSolutions;
	CUDAPointer<char>   relDirections;
	CUDAPointer<char>   moreRelDirections;
	CUDAPointer<int>    contacts;
	CUDAPointer<int>    bestContact;
	CUDAPointer<char>   hpChain;
};
