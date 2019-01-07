#include <iostream>

#include "acopredictor.h"

using std::cout;
using std::cerr;

ACOPredictor::ACOPredictor(const HPChain &hpchain)
: dhpchain(hpchain),
  dPheromone( new double[(hpchain.length()-2)*5]() ) /* Value-initialized to 0.0 */ {
	for(int j = 0; j < 5; j++){
		for(int i = 0; i < hpchain.length()-2; i++){
			cout << pheromone(i, j) << " ";
		} cout << "\n";
	}
}

ACOPredictor::~ACOPredictor(){
	delete[] dPheromone;
}

inline double ACOPredictor::pheromone(int i, int d) const {
	return dPheromone[i*5 + d];
}

MovChain ACOPredictor::predict(){
	return MovChain();
}
