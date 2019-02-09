#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <cstring>
#include <memory>
#include <mpi.h>

#include "acopredictor.h"

using std::cout;
using std::cerr;
using std::vector;
using std::priority_queue;
using std::string;
using std::unique_ptr;
using std::pair;
using std::make_pair;

void *serialize_solution(const vector<char> &directions, int nContacts, int &resultingBufferSize){
	int nDirections = directions.size();

	int bufferSize = 0;
	bufferSize += sizeof(nDirections);
	bufferSize += sizeof(nContacts);
	bufferSize += sizeof(char) * nDirections;

	// Will contain [nDirections] [nContacts] [direction1, direction2, ..., directionN]
	void *buffer = (void *) new char[bufferSize];
	
	int *intBuffer = (int*) buffer;
	*intBuffer = nDirections;
	intBuffer++;
	*intBuffer = nContacts;
	intBuffer++;

	char *charBuffer = (char*) intBuffer;
	memcpy(charBuffer, directions.data(), sizeof(char) * nDirections);

	// At this point, 'buffer' is filled

	resultingBufferSize = bufferSize;
	return buffer;
}

vector<char> deserialize_solution(void *bundle, int &nContacts){
	int *intPointer = (int *) bundle;

	int nDirections = *intPointer;
	intPointer++;
	nContacts = *intPointer;
	intPointer++;

	char *charPointer = (char *) intPointer;

	vector<char> directions;
	directions.reserve(nDirections);
	for(int i = 0; i < nDirections; i++){
		directions.push_back(*charPointer);
		charPointer += 1;
	}

	return directions;
}

void send_solution(const vector<char> &directions, int contacts, int destIdx){
	int bufferSize;
	void *sendBuffer = serialize_solution(directions, contacts, bufferSize);
	MPI_Send(sendBuffer, bufferSize, MPI_CHAR, destIdx, 0, MPI_COMM_WORLD);
	delete[] (char *) sendBuffer;
}

void recv_solution(vector<char> &directions, int &contacts, int recvBufSize, int srcIdx){
	void *recvBuffer = (void *) new char[recvBufSize];
	MPI_Recv(recvBuffer, recvBufSize, MPI_CHAR, srcIdx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	directions = deserialize_solution(recvBuffer, contacts);
}

void ring_exchange(
		const vector<char> &localDirections,
		int localContacts,
		vector<char> &receivedDirections,
		int &receivedNContacts)
{

}

struct ACOPredictor::Results ACOPredictor::predict(){
	MPI_Init(NULL, NULL);

	int myRank, commSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);

	// Since we are running multiple processes, we have to guarantee they use different random seeds
	if(dRandSeed < 0){
		std::random_device rd;
		dRandGen.seed(rd() + myRank);
	} else {
		dRandGen.seed(dRandSeed + myRank);
	}

	if(commSize == 1){
		cerr << "Ran multi-colony program with only one node! This is not allowed.\n";
		exit(EXIT_FAILURE);
	}

	if(dExchangedAnts <= 0){
		cerr << "Ran multi-colony program but specified weird number of EXCHANGED ANTS per cycle. Please check the configuration file.\n";
		exit(EXIT_FAILURE);
	}

	ACOSolution bestSol;
	int bestContacts = -1;

	for(int i = 0; i < dCycles; i++){
		vector<ACOSolution> antsSolutions; // Solutions generated by all ants

		// Let each ant develop a solution
		for(int j = 0; j < dNAnts; j++){
			ACOSolution currentSol = ant_develop_solution();
			if(currentSol.has_error() == false){
				antsSolutions.push_back(currentSol);
			}
		}

		// Calculate contacts
		unique_ptr<int[]> nContacts(new int[antsSolutions.size()]);
		for(unsigned j = 0; j < antsSolutions.size(); j++)
			nContacts[j] = antsSolutions[j].count_contacts(dHPChain);

		// Perform local search
		for(unsigned j = 0; j < antsSolutions.size(); j++){
			ACOSolution tentative = antsSolutions[j];
			tentative.perturb_one(dRandGen);
			int contacts = tentative.count_contacts(dHPChain);
			if(contacts > nContacts[j]){
				antsSolutions[j] = tentative;
			}
		}

		// Check best protein
		for(unsigned j = 0; j < antsSolutions.size(); j++){
			if(nContacts[j] > bestContacts){
				bestSol = antsSolutions[j];
				bestContacts = nContacts[j];
			}
		}

		// Deposit pheromones
		for(unsigned j = 0; j < antsSolutions.size(); j++)
			ant_deposit_pheromone(antsSolutions[j].directions(), nContacts[j]);

		// We will need to select the best proteins of current cycle, so we place them all in a priority queue
		auto cmp = [](pair<int,ACOSolution&> a, pair<int,ACOSolution&> b){ return a.first < b.first; };
		priority_queue<pair<int,ACOSolution&>, vector<pair<int,ACOSolution&>>, decltype(cmp)> que(cmp);
		for(unsigned int j = 0; j < antsSolutions.size(); j++)
			que.push(pair<int,ACOSolution&>(nContacts[j], antsSolutions[j]));

		// Exchange N solutions with other colonies
		for(int j = 0; j < dExchangedAnts; j++){
			vector<char> receivedDirections;
			int receivedNContacts;
			vector<char> sendDirections;
			int sendNContacts;

			if(j == 0){
				// On first iteration we exchange globally best solutions
				sendDirections = bestSol.directions();
				sendNContacts = bestContacts;
			} else {
				// On other iterations, we exchange best solutions within the pool generated in the current cycle
				pair<int,ACOSolution&> p = que.top();
				que.pop();
				sendDirections = p.second.directions();
				sendNContacts = p.first;
			}

			// Perform ring-exchange
			int recvBufSize = sizeof(int) + sizeof(int) + sizeof(char) * bestSol.directions().size();
			if(myRank%2 == 0){
				send_solution(sendDirections, sendNContacts, (myRank+1)%commSize);
				recv_solution(receivedDirections, receivedNContacts, recvBufSize, (myRank-1+commSize)%commSize);
			} else {
				recv_solution(receivedDirections, receivedNContacts, recvBufSize, (myRank-1+commSize)%commSize);
				send_solution(sendDirections, sendNContacts, (myRank+1)%commSize);
			}

			//cout << "I am " << myRank << ", sent " << sendNContacts << " and received " << receivedNContacts << "\n";

			// Deposit pheromones for received ant
			ant_deposit_pheromone(receivedDirections, receivedNContacts);
		}

		/*
		if(myRank == 0){
			cout << receivedNContacts << ": ";
			for(char c: receivedDirections){
				cout << (int) c << " ";
			} cout << "\n";
		}
		*/

		// Evaporate pheromones
		evaporate_pheromone();

		/*
		if(i%2 == 0){
			for(int j = 0; j < 5; j++){
				for(int i = 0; i < dNMovElems; i++)
					cout << pheromone(i, j) << " ";
				cout << "\n";
			}
			cout << "\n";
		}
		*/

		// if(i%5 == 0) cout << "Cycle: " << i << "\n";
	}

	// Gather solutions in node 0
	if(myRank == 0){
		int recvBufSize = sizeof(int) + sizeof(int) + sizeof(char) * bestSol.directions().size();
		vector<char> receivedDirections;
		int receivedNContacts;

		//cout << "Solution from node 0 is " << bestContacts << ".\n";

		// Check which one is best
		for(int i = 1; i < commSize; i++){
			recv_solution(receivedDirections, receivedNContacts, recvBufSize, i);
			//cout << "Solution from node " << i << " is " << receivedNContacts << ".\n";
			if(receivedNContacts > bestContacts){
				bestSol = ACOSolution(receivedDirections);
				bestContacts = receivedNContacts;
			}
		}
	} else {
		send_solution(bestSol.directions(), bestContacts, 0);
	}

	Results res = {
		.solution = bestSol,
		.contacts = bestContacts
	};

	MPI_Finalize();

	if(myRank != 0){
		exit(EXIT_SUCCESS);
	}

	return res;
}
