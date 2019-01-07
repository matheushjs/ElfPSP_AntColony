#include <iostream>
#include <cstdlib>
#include <ctime>

#include "hpchain.h"
#include "movchain.h"
#include "config.h"
#include "vec3.h"
#include "acopredictor.h"

using namespace std;

int main(int argc, char *argv[]){
	Config conf;
	HPChain hpchain(conf.hp_chain());

	if(!hpchain.validate()){
		cerr << "Invallid HP Chain. Received: " << hpchain.get_chain() << "\n";
		std::exit(1);
	}

	if(conf.random_seed() < 0){
		srand(time(NULL));
	} else {
		srand(conf.random_seed());
	}

	ACOPredictor predictor(hpchain, conf);
	MovChain result = predictor.predict();

	cout << result.get_chain() << "\n";

	return 0;
}
