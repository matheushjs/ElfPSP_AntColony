#include <iostream>
#include <ctime>

#include "hpchain.h"
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

	ACOPredictor predictor(
		conf.hp_chain(),
		conf.cycles(),
		conf.n_ants(),
		conf.aco_alpha(),
		conf.aco_beta(),
		conf.aco_evaporation(),
		conf.random_seed()
	);

	ACOSolution result = predictor.predict();

	cout << result << "\n";

	return 0;
}
