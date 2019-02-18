#pragma once

#include <string>

#include "cuda_device_utilities.cuh"

/** Holds data needed by each CUDA thread to execute its work.
 * We associate __device__ functions with this structure, which makes access to this
 *   struct's data by a thread much easier.
 */
struct CUDAThread {
	int tid; /**< Unique identifier for the thread within all blocks (the grid). */
	int randNumber; /**< Random number that also serves as the seed for generation. */
	int3 *mySolution; /**< Memory region containing coordinates of the solution this
	                       thread is developing. */
	int3 *myOtherSolution; /**< Memory region that the thread uses for performing
	                            local seach (small perturbation) on mySolution. */
	char *myDirections; /**< Memory containing relative directions for mySolution. */
	char *myOtherDirections; /**< Memory containing relative directions for myOtherSolution. */
	double *pheromones; /**< Pointer to the pheromone matrix (shared by all threads). */
	char *hpChain; /**< hpChain of the protein being predicted (shared by all threads). */
	int nCoords; /**< Number of coordinates of the protein. */
	int nMovElems; /**< Number of directions of the protein. */
	double dAlpha; /**< Alpha parameter of the ACO algorithm. */
	double dBeta; /**< Beta parameter of the ACO algorithm. */

	static __device__ int3 DIRECTION_VECTOR(int3 prevDir, char dir);

	__device__ void solution_from_directions(int3 *solution, char *directions);
	__device__ void local_search(int &solContact, int lsFreq);

	__device__ double &pheromone(int i, int d);
	__device__ int calculate_contacts(int3 *solution);
	__device__ void get_heuristics(int curSize, int3 *solution, double *heurs, int3 *possiblePos);
	__device__ void get_probabilities(int movIndex, double *probs, double *heurs);
	__device__ void develop_solution(int3 *solution, char *directions);
};

namespace HostToDevice {
	__global__
	void ant_develop_solution(
		double *pheromone,   int nMovElems,
		int3 *solutions,     int3 *moreSolutions, int nCoords,
		char *relDirections, char *moreRelDirections,
		int *contacts,       char *hpChain,       int lsFreq,
		double alpha,        double beta
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
